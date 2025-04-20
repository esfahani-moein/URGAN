print("Script initialization started...")
import os
import random
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
print("Importing Torch...")
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.amp import GradScaler
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel

print("importing Loader and Model...")

# Import project modules
import models as model
from data_fetch import load_dataset

print("Importing utils...")

from utils import CheckpointManager, save_checkpoint, load_checkpoint
from utils import build_iqa_model, Summary, AverageMeter, ProgressMeter
from utils import create_logger, log_metrics, log_training,set_distributed_GPU

from configs import config_model_net as config
from configs import config_dataset


def main():
    """Main training function for ESRGAN."""
    
    print("Starting training...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set fixed random seeds fro reproducability
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # Use cudnn benchmark for fixed-size inputs
    cudnn.benchmark = True
    
    # Initialize mixed precision with bfloat16 for H100 GPUs
    scaler = GradScaler('cuda')
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Default to start training from scratch
    start_epoch = 0
    
    # Initialize best metrics
    best_psnr = 0.0
    best_ssim = 0.0
    
    # Set up distributed training if multiple GPUs available
    print("Set Distributed Training...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # distributed = torch.cuda.device_count() > 1
    distributed = False
    
    device, is_main_process = set_distributed_GPU(distributed)
    
    print(f"Using device: {device}")
    
    print("Loading Dataset...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Create data loaders
    train_dataloader, test_dataloader = load_dataset(config_dataset, device, distributed)
    
    print("Building or Compiling the Model...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Build model
    g_model, ema_g_model = build_model(config, device)
    
    print("Define Evaluation Functions...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define loss function
    pixel_criterion = define_loss(config, device)
    
    # Define optimizer
    optimizer = define_optimizer(g_model, config)
    
    # Define learning rate scheduler
    scheduler = define_scheduler(optimizer, config)
    
    print("Checkpoints Loading...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize checkpoint manager
    if is_main_process:  # Only the main process saves checkpoints
        checkpoint_dir = os.path.join("checkpoints", config.exp_name)
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=config.checkpoint.max_checkpoints,
            use_safetensors=True  # Use SafeTensors for H100 optimization
        )
    
    # Load pretrained or resume from checkpoint
    if is_main_process and config.checkpoint.pretrained_g_model:
        checkpoint_info = load_checkpoint(
            checkpoint_manager=checkpoint_manager,
            path=config.checkpoint.pretrained_g_model,
            model=g_model,
            device=device
        )
        print(f"Loaded pretrained model: {config.checkpoint.pretrained_g_model}")
    
    if is_main_process and config.checkpoint.resumed_g_model:
        checkpoint_info = load_checkpoint(
            checkpoint_manager=checkpoint_manager,
            path=config.checkpoint.resumed_g_model,
            model=g_model,
            ema_model=ema_g_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = checkpoint_info.get("epoch", 0)
        best_psnr = checkpoint_info.get("psnr", 0.0)
        best_ssim = checkpoint_info.get("ssim", 0.0)
        print(f"Resumed from epoch {start_epoch}, PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f}")
    
    # Set up metrics models for test evaluation
    psnr_model, ssim_model = build_iqa_model(
        scale=config.scale,
        only_test_y_channel=config.only_test_y_channel,
        device=device
    )
    
    # Create directories for logs and samples
    if is_main_process:
        # samples_dir = os.path.join("reports", config.exp_name)
        # os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(config.exp_name)
        log_training(logger['txt_path'], f"Training started - Epochs: {config.epochs}, Batch size: {config_dataset.batch_size}")
        log_training(logger['txt_path'], f"Model: {config.model.g_name}, Optimizer: {config.optimizer.name}, LR: {config.optimizer.lr}")
    

    print("Starting training Loop...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Training loop
    print(f"Starting training for {config.epochs} epochs")

    for epoch in range(start_epoch, config.epochs):
        # Set epoch for distributed sampler
        if distributed and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        


        if is_main_process:
                start_mem = torch.cuda.max_memory_allocated() / 1024**3
                log_training(logger['txt_path'], f"Starting epoch memory: {start_mem:.2f} GB")
    


        # Train for one epoch
        train_loss = train_epoch(
            g_model=g_model,
            ema_g_model=ema_g_model,
            train_dataloader=train_dataloader,
            pixel_criterion=pixel_criterion,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            logger=logger if is_main_process else None,
            device=device,
            config=config,
            is_main_process=is_main_process
        )
        


        if is_main_process:
                end_mem = torch.cuda.max_memory_allocated() / 1024**3
                log_training(logger['txt_path'], f"Peak training memory: {end_mem:.2f} GB")


        # Update learning rate
        scheduler.step()
        
        # Evaluate model (only on main process or if not distributed)
        if is_main_process or not distributed:
            psnr, ssim = test(
                model=g_model,
                test_dataloader=test_dataloader,
                psnr_model=psnr_model,
                ssim_model=ssim_model,
                device=device
            )
            print(f"\n[Epoch {epoch+1}] PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")
            
            # Save metrics to TensorBoard
            if is_main_process:
                current_lr = optimizer.param_groups[0]["lr"]
                log_metrics(logger['csv_path'], epoch + 1, psnr, ssim, train_loss, current_lr)
                log_training(logger['txt_path'], f"Epoch {epoch+1}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, Loss={train_loss:.6f}, LR={current_lr:.8f}")
            
            # Check for best model
            is_best = psnr > best_psnr and ssim > best_ssim
            best_psnr = max(psnr, best_psnr)
            best_ssim = max(ssim, best_ssim)
            
            # Save checkpoint periodically and for best model
            if is_main_process and ((epoch + 1) % config.checkpoint.save_freq == 0 or is_best):
                save_checkpoint(
                    checkpoint_manager=checkpoint_manager,
                    model=g_model,
                    ema_model=ema_g_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    psnr=psnr,
                    ssim=ssim,
                    is_best=is_best,
                    name=f"{config.exp_name}"
                )
    
    # Final checkpoint
    if is_main_process:
        save_checkpoint(
            checkpoint_manager=checkpoint_manager,
            model=g_model,
            ema_model=ema_g_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config.epochs,
            psnr=best_psnr,
            ssim=best_ssim,
            is_best=False,
            name=f"{config.exp_name}_final"
        )
        
        log_training(logger['txt_path'], f"Training completed. Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}")
        print("Training completed.")





def build_model(config, device):
    """Build generator model with the same architecture as original.
    
    Args:
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        Tuple of (generator_model, ema_model)
    """
    # Create generator model
    g_model = model.__dict__[config.model.g_name](
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        channels=config.model.channels,
        growth_channels=config.model.growth_channels,
        num_rrdb=config.model.num_rrdb
    )
    g_model = g_model.to(device)
    
    # Create EMA model if enabled
    if config.model.ema.enable:
        ema_decay = config.model.ema.decay
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_g_model = AveragedModel(g_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_g_model = None
    
    # Use torch.compile if enabled and available (PyTorch 2.0+)
    if config.model.compile and hasattr(torch, "compile"):
        try:
            g_model = torch.compile(g_model, mode="max-autotune")
            print("Using torch.compile for generator model")
        except Exception as e:
            print(f"Failed to compile model: {e}")
    
    return g_model, ema_g_model


def define_loss(config, device):
    """Define loss function based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        Loss function
    """
    if config.pixel_loss.name == "L1Loss":
        pixel_criterion = nn.L1Loss()
    elif config.pixel_loss.name == "MSELoss":
        pixel_criterion = nn.MSELoss()
    elif config.pixel_loss.name == "SmoothL1Loss":
        pixel_criterion = nn.SmoothL1Loss()
    else:
        raise NotImplementedError(f"Loss {config['pixel_loss']['name']} is not implemented")
    
    pixel_criterion = pixel_criterion.to(device)
    return pixel_criterion


def define_optimizer(g_model, config):
    """Define optimizer based on configuration.
    
    Args:
        g_model: Generator model
        config: Configuration dictionary
        
    Returns:
        Optimizer
    """
    if config.optimizer.name == "Adam":
        optimizer = optim.Adam(
            g_model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay
        )
    elif config.optimizer.name == "AdamW":
        optimizer = optim.AdamW(
            g_model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer.name} is not implemented")
    
    return optimizer


def define_scheduler(optimizer, config):
    """Define learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer
        config: Configuration dictionary
        
    Returns:
        Learning rate scheduler
    """
    if config.lr_scheduler.name == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_scheduler.step_size,
            gamma=config.lr_scheduler.gamma
        )
    elif config.lr_scheduler.name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.lr_scheduler.get("T_max", 200000),
            eta_min=config.lr_scheduler.get("eta_min", 0)
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(
    g_model,
    ema_g_model,
    train_dataloader,
    pixel_criterion,
    optimizer,
    epoch,
    scaler,
    logger,
    device,
    config,
    is_main_process=True
):
    """Train model for one epoch with H100 optimizations.
    
    Args:
        g_model: Generator model
        ema_g_model: EMA model (can be None)
        train_dataloader: Training data loader
        pixel_criterion: Loss function
        optimizer: Optimizer
        epoch: Current epoch number
        scaler: Gradient scaler for mixed precision
        writer: TensorBoard writer
        device: Device to use
        config: Configuration dictionary
        is_main_process: Whether this is the main process (for distributed training)
    """
    # Calculate how many batches of data are in each epoch
    batches = len(train_dataloader)
    
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(
        batches,
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch + 1}]"
    )
    
    # Put the model in training mode
    g_model.train()
    
    # Get loss weights
    loss_weight = torch.Tensor(config.pixel_loss.weight).to(device)
    
    # Get gradient accumulation steps
    accumulation_steps = config.get("gradient_accumulation_steps", 1)
    
    # Start timing
    end = time.time()
    
    # Zero gradients at the beginning
    optimizer.zero_grad(set_to_none=True)
    
    for batch_index, batch_data in enumerate(train_dataloader):
        # Move data to device efficiently
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)
        
        # Record data loading time
        data_time.update(time.time() - end)
        
        # Use bfloat16
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Forward pass
            sr = g_model(lr)
            pixel_loss = pixel_criterion(sr, gt)
            pixel_loss = torch.sum(torch.mul(loss_weight, pixel_loss))
            
            # Normalize loss for gradient accumulation
            if accumulation_steps > 1:
                pixel_loss = pixel_loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(pixel_loss).backward()
        
        # Update weights after accumulation steps or on last batch
        if ((batch_index + 1) % accumulation_steps == 0) or (batch_index + 1 == batches):
            # Gradient clipping for stability
            if config.get("gradient_clip_norm", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    g_model.parameters(),
                    config["gradient_clip_norm"]
                )
            
            # Step optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA model if enabled
            if config.model.ema.enable and ema_g_model is not None:
                ema_g_model.update_parameters(g_model)
        
        # Record loss
        losses.update(pixel_loss.item() * (1 if accumulation_steps <= 1 else accumulation_steps), lr.size(0))
        
        # Record batch time
        batch_time.update(time.time() - end)
        end = time.time()
    

        # Output training logs
        if is_main_process and batch_index % config.print_freq == 0:
            progress_msg = f"Epoch: [{epoch+1}][{batch_index}/{batches}] Time {batch_time.val:.3f} ({batch_time.avg:.3f}) Loss {losses.val:.6f} ({losses.avg:.6f})"
            
            if logger is not None:
                log_training(logger['txt_path'], progress_msg)
            
            # Print progress
            progress.display(batch_index)

    final_loss = losses.avg
    if is_main_process and logger is not None:
        summary_msg = f"Epoch: [{epoch+1}] completed. Time {batch_time.avg:.3f} Loss {losses.avg:.6f}"
        log_training(logger['txt_path'], summary_msg)
    
    return final_loss




def test(model, test_dataloader, psnr_model, ssim_model, device):
    """Test the model and compute PSNR and SSIM.
    
    Args:
        model: Generator model
        test_dataloader: Test data loader
        psnr_model: PSNR calculator model
        ssim_model: SSIM calculator model
        device: Device to use
        
    Returns:
        Tuple of (average PSNR, average SSIM)
    """
    # Ensure model is in eval mode
    model.eval()
    
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for batch_data in test_dataloader:
            gt = batch_data["gt"].to(device)
            lr = batch_data["lr"].to(device)
            
            # Process in smaller chunks if batch size > 1
            sub_batch_size = 1  # Process one image at a time
            for i in range(0, gt.size(0), sub_batch_size):
                gt_sub = gt[i:i+sub_batch_size]
                lr_sub = lr[i:i+sub_batch_size]
                
                sr_sub = model(lr_sub)
                
                # Calculate metrics
                batch_psnr = psnr_model(sr_sub, gt_sub)
                batch_ssim = ssim_model(sr_sub, gt_sub)
                
                # Collect metrics
                psnr_list.extend(batch_psnr.tolist())
                ssim_list.extend(batch_ssim.tolist())
                
                # Explicitly delete tensors to free memory
                del sr_sub, gt_sub, lr_sub
            
            # Clear batch data
            del gt, lr
            torch.cuda.empty_cache()  # Force CUDA memory cleanup
    
    # Calculate average metrics
    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0
    
    return avg_psnr, avg_ssim


if __name__ == "__main__":

    print("Script wants to begin main function...")

    main()