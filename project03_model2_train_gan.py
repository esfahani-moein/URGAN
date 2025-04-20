print("GAN Super Resolution Script initialization started...")
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import os
import random
import time
import csv
import numpy as np

print("Importing Torch...")
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.amp import GradScaler
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel

print("Importing Loader and Model...")
import models as model
from data_fetch import load_dataset
# from imgproc import apply_augmentations

print("Importing utils...")
import utils as u

from configs import config_model_gan as config
from configs import config_dataset


def main():
    """Main training function for GAN-based Super Resolution."""
    
    print("Starting GAN training...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Fixed random number seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True
    
    # Initialize mixed precision with bfloat16 for H100 GPUs
    scaler = GradScaler('cuda')
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True

    # Default to start training from scratch
    start_epoch = 0

    # Initialize the image clarity evaluation index
    best_psnr = 0.0
    best_ssim = 0.0

    # Set up distributed training if multiple GPUs available
    print("Set Distributed Training...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # distributed = torch.cuda.device_count() > 1
    distributed = False
    
    device, is_main_process = u.set_distributed_GPU(distributed)
    
    print(f"Using device: {device}")
    
    print("Loading Dataset...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Create data loaders
    train_dataloader, test_dataloader = load_dataset(config_dataset, device, distributed)
    
    print("Building or Compiling the Model...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Build generator and discriminator models
    g_model, ema_g_model, d_model = build_model(config, device)
    
    print("Define Loss Functions...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Define loss functions
    pixel_criterion, feature_criterion, adversarial_criterion = define_loss(config, device)
    
    print("Define Optimizers...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Define optimizers
    g_optimizer, d_optimizer = define_optimizer(g_model, d_model, config)
    
    print("Define Learning Rate Schedulers...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Define learning rate schedulers
    g_scheduler, d_scheduler = define_scheduler(g_optimizer, d_optimizer, config)
    
    print("Checkpoints Loading...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize checkpoint manager for GAN training
    if is_main_process:  # Only the main process saves checkpoints
        g_checkpoint_manager, d_checkpoint_manager = u.load_latest_checkpoints(config,device,                                   
                                                                                g_model,d_model,g_optimizer,g_scheduler,d_optimizer,d_scheduler,ema_g_model)

    # Set up metrics models for test evaluation
    psnr_model, ssim_model = u.build_iqa_model(
        scale=config.scale,
        only_test_y_channel=config.test.only_test_y_channel,
        device=device
    )
    
    # Create directories for logs and results
    if is_main_process:
        results_dir = os.path.join("results", config.exp_name)
        os.makedirs(results_dir, exist_ok=True)
        
        logger = u.create_logger(config.exp_name)
        u.log_training(logger['txt_path'], f"GAN Training started - Epochs: {config.train.hyp.epochs}, Batch size: {config_dataset.batch_size}")
        u.log_training(logger['txt_path'], f"Generator: {config.model.g.name}, Discriminator: {config.model.d.name}")
        u.log_training(logger['txt_path'], f"Optimizer: {config.train.optim.name}, LR: {config.train.optim.lr}")
        
        # Create CSV file for logging metrics
        csv_path = os.path.join("reports", f"{config.exp_name}_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'G_Loss', 'D_Loss', 'Pixel_Loss', 'Content_Loss', 'Adversarial_Loss', 'PSNR', 'SSIM', 'Learning_Rate'])

    print("Starting training Loop...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    # Training loop
    print(f"Starting training for {config.train.hyp.epochs} epochs")

    for epoch in range(start_epoch, config.train.hyp.epochs):
        # Set epoch for distributed sampler
        if distributed and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        # Train for one epoch
        train_metrics = model.train_gan(
            g_model=g_model,
            ema_g_model=ema_g_model,
            d_model=d_model,
            train_dataloader=train_dataloader,
            pixel_criterion=pixel_criterion,
            feature_criterion=feature_criterion,
            adversarial_criterion=adversarial_criterion,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            epoch=epoch,
            scaler=scaler,
            logger=logger if is_main_process else None,
            device=device,
            config=config,
            is_main_process=is_main_process
        )

        # Update learning rate
        g_scheduler.step()
        d_scheduler.step()

        # Evaluate model (only on main process or if not distributed)
        if is_main_process or not distributed:
            # Use EMA model for evaluation if available
            eval_model = ema_g_model if ema_g_model is not None else g_model
            
            psnr, ssim = u.model_test(
                model=eval_model,
                test_dataloader=test_dataloader,
                psnr_model=psnr_model,
                ssim_model=ssim_model,
                device=device
            )
            
            print(f"\nEvaluation - PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
            
            # Log metrics
            if is_main_process:
                u.log_training(logger['txt_path'], 
                             f"Epoch {epoch+1}/{config.train.hyp.epochs} - "
                             f"G_Loss: {train_metrics['g_loss']:.6f}, "
                             f"D_Loss: {train_metrics['d_loss']:.6f}, "
                             f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
                
                # Write to CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1, 
                        train_metrics['g_loss'],
                        train_metrics['d_loss'],
                        train_metrics['pixel_loss'],
                        train_metrics['content_loss'],
                        train_metrics['adversarial_loss'],
                        psnr,
                        ssim,
                        g_optimizer.param_groups[0]['lr']
                    ])
                
                # Check for best model and save checkpoints
                is_best = psnr > best_psnr and ssim > best_ssim
                is_last = (epoch + 1) == config.train.hyp.epochs
                best_psnr = max(psnr, best_psnr)
                best_ssim = max(ssim, best_ssim)
                
                # Save generator checkpoint
                u.save_checkpoint(
                    checkpoint_manager=g_checkpoint_manager,
                    model=g_model,
                    ema_model=ema_g_model,
                    optimizer=g_optimizer,
                    scheduler=g_scheduler,
                    epoch=epoch + 1,
                    # metrics={"psnr": psnr, "ssim": ssim},
                    is_best=is_best
                )
                
                # Save discriminator checkpoint
                u.save_checkpoint(
                    checkpoint_manager=d_checkpoint_manager,
                    model=d_model,
                    optimizer=d_optimizer,
                    scheduler=d_scheduler,
                    epoch=epoch + 1,
                    # metrics={"psnr": psnr, "ssim": ssim},
                    is_best=is_best
                )
    
    # Final log entry
    if is_main_process:
        u.log_training(logger['txt_path'], f"Training completed. Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}")
        print(f"Training completed. Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}")


def build_model(config, device):
    """Build generator and discriminator models.
    
    Args:
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        Tuple of (generator_model, ema_model, discriminator_model)
    """
    # Create generator model
    g_model = model.__dict__[config.model.g.name](
        in_channels=config.model.g.in_channels,
        out_channels=config.model.g.out_channels,
        channels=config.model.g.channels,
        growth_channels=config.model.g.growth_channels,
        num_rrdb=config.model.g.num_rrdb
    )
    
    # Create discriminator model
    d_model = model.__dict__[config.model.d.name](
        in_channels=config.model.d.in_channels,
        out_channels=config.model.d.out_channels,
        channels=config.model.d.channels     
    )
    
    # Move models to device
    g_model = g_model.to(device)
    d_model = d_model.to(device)
    
    # Optional: For H100 GPU optimization
    g_model = g_model.to(memory_format=torch.channels_last)
    d_model = d_model.to(memory_format=torch.channels_last)
    
    # Create EMA model if enabled
    if config.model.ema.enable:
        # Generate an exponential average model based on a generator to stabilize model training
        ema_decay = config.model.ema.decay
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_g_model = AveragedModel(g_model, device=device, avg_fn=ema_avg_fn)
        # Optional: For H100 GPU optimization
        ema_g_model = ema_g_model.to(memory_format=torch.channels_last)
    else:
        ema_g_model = None
    
    # # Use torch.compile if enabled (PyTorch 2.0+)
    # if config.model.g.compiled and hasattr(torch, "compile"):
    #     g_model = torch.compile(g_model, mode="reduce-overhead")
    # if config.model.d.compiled and hasattr(torch, "compile"):
    #     d_model = torch.compile(d_model, mode="reduce-overhead")
    # if config.model.ema.compiled and ema_g_model is not None and hasattr(torch, "compile"):
    #     ema_g_model = torch.compile(ema_g_model, mode="reduce-overhead")
    
    return g_model, ema_g_model, d_model


def define_loss(config, device):
    """Define loss functions based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        Tuple of (pixel_criterion, feature_criterion, adversarial_criterion)
    """
    # Pixel loss
    if config.train.losses.pixel_loss.name == "L1Loss":
        pixel_criterion = nn.L1Loss()
    else:
        raise NotImplementedError(f"Loss {config.train.losses.pixel_loss.name} is not implemented.")
    
    # Content/Feature loss
    if config.train.losses.content_loss.name == "ContentLoss":
        # Ensure VGG model path exists
        vgg_weights_path = config.train.losses.content_loss.model_weights_path
        if vgg_weights_path and os.path.exists(vgg_weights_path):
            print(f"Loading VGG19 weights from: {vgg_weights_path}")
        else:
            print(f"WARNING: VGG19 weights not found at {vgg_weights_path}. Using default weights.")
            

        feature_criterion = model.ContentLoss(
            config.train.losses.content_loss.net_cfg_name,
            config.train.losses.content_loss.batch_norm,
            config.train.losses.content_loss.num_classes,
            config.train.losses.content_loss.model_weights_path,
            config.train.losses.content_loss.feature_nodes,
            config.train.losses.content_loss.feature_normalize_mean,
            config.train.losses.content_loss.feature_normalize_std,
        )
    else:
        raise NotImplementedError(f"Loss {config.train.losses.content_loss.name} is not implemented.")
    
    # Adversarial loss
    if config.train.losses.adversarial_loss.name == "vanilla":
        adversarial_criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Loss {config.train.losses.adversarial_loss.name} is not implemented.")
    
    # Move loss functions to device
    pixel_criterion = pixel_criterion.to(device)
    feature_criterion = feature_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)
    
    return pixel_criterion, feature_criterion, adversarial_criterion


def define_optimizer(g_model, d_model, config):
    """Define optimizers based on configuration.
    
    Args:
        g_model: Generator model
        d_model: Discriminator model
        config: Configuration dictionary
        
    Returns:
        Tuple of (generator_optimizer, discriminator_optimizer)
    """
    if config.train.optim.name == "Adam":
        g_optimizer = optim.Adam(
            g_model.parameters(),
            lr=config.train.optim.lr,
            betas=config.train.optim.betas,
            eps=config.train.optim.eps,
            weight_decay=config.train.optim.weight_decay
        )
        d_optimizer = optim.Adam(
            d_model.parameters(),
            lr=config.train.optim.lr,
            betas=config.train.optim.betas,
            eps=config.train.optim.eps,
            weight_decay=config.train.optim.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {config.train.optim.name} is not implemented.")
    
    return g_optimizer, d_optimizer


def define_scheduler(g_optimizer, d_optimizer, config):
    """Define learning rate schedulers based on configuration.
    
    Args:
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        config: Configuration dictionary
        
    Returns:
        Tuple of (generator_scheduler, discriminator_scheduler)
    """
    if config.train.lr_scheduler.name == "MultiStepLR":
        g_scheduler = lr_scheduler.MultiStepLR(
            g_optimizer,
            milestones=config.train.lr_scheduler.milestones,
            gamma=config.train.lr_scheduler.gamma
        )
        d_scheduler = lr_scheduler.MultiStepLR(
            d_optimizer,
            milestones=config.train.lr_scheduler.milestones,
            gamma=config.train.lr_scheduler.gamma
        )
    else:
        raise NotImplementedError(f"LR Scheduler {config.train.lr_scheduler.name} is not implemented.")
    
    return g_scheduler, d_scheduler




if __name__ == "__main__":
    print("Script wants to begin main function...")
    main()