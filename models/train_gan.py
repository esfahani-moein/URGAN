import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils as u


def train_gan(
    g_model,
    ema_g_model,
    d_model,
    train_dataloader,
    pixel_criterion,
    feature_criterion,
    adversarial_criterion,
    g_optimizer,
    d_optimizer,
    epoch,
    scaler,
    logger,
    device,
    config,
    is_main_process=True
):
    """Train GAN models for one epoch with optimizations for modern GPUs.
    
    Args:
        g_model: Generator model
        ema_g_model: EMA model (can be None)
        d_model: Discriminator model
        train_dataloader: Training data loader
        pixel_criterion: Pixel loss function
        feature_criterion: Feature/content loss function
        adversarial_criterion: Adversarial loss function
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        epoch: Current epoch number
        scaler: Gradient scaler for mixed precision
        logger: Logger for recording metrics
        device: Device to use
        config: Configuration dictionary
        is_main_process: Whether this is the main process (for distributed training)
        
    Returns:
        Dictionary of average metrics for the epoch
    """
    # Calculate how many batches of data are in each epoch
    batches = len(train_dataloader)
    
    # Print information of progress bar during training
    batch_time = u.AverageMeter("Time", ":6.3f", u.Summary.NONE)
    data_time = u.AverageMeter("Data", ":6.3f", u.Summary.NONE)
    g_losses = u.AverageMeter("G Loss", ":6.6f", u.Summary.NONE)
    d_losses = u.AverageMeter("D Loss", ":6.6f", u.Summary.NONE)
    pixel_losses = u.AverageMeter("Pixel Loss", ":6.6f", u.Summary.NONE)
    content_losses = u.AverageMeter("Content Loss", ":6.6f", u.Summary.NONE)
    adversarial_losses = u.AverageMeter("Adv Loss", ":6.6f", u.Summary.NONE)
    
    progress = u.ProgressMeter(
        batches,
        [batch_time, data_time, g_losses, d_losses],
        prefix=f"Epoch: [{epoch + 1}]"
    )
    
    # Set models to training mode
    g_model.train()
    d_model.train()
    
    # Define loss function weights
    pixel_weight = torch.Tensor(config.train.losses.pixel_loss.weight).to(device)
    content_weight = torch.Tensor(config.train.losses.content_loss.weight).to(device)
    adversarial_weight = torch.Tensor(config.train.losses.adversarial_loss.weight).to(device)
    
    # Initialize batch timing
    end = time.time()
    
    # Check if we should use CUDA graph marking
    # has_cuda_graph_marking = hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin")
    has_cuda_graph_marking = False

    # Training loop
    for batch_index, batch_data in enumerate(train_dataloader):
        # Get data and move to device
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)
        
        # Record data loading time
        data_time.update(time.time() - end)
        
        # Get batch size
        batch_size = gt.shape[0]
        
        # Apply data augmentation
        gt, lr = u.apply_augmentations(gt, lr, config)
        
        # Set up labels for adversarial loss
        if config.model.d.name == "discriminator_for_vgg":
            real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=device)
            fake_label = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=device)
        else:
            raise ValueError(f"The discriminator {config.model.d.name} is not supported.")
        
        #--------------------------------
        # Train Generator
        #--------------------------------
        # Disable discriminator gradients during generator training
        for d_param in d_model.parameters():
            d_param.requires_grad = False
        
        # Zero generator gradients
        g_optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Generate super-resolution image
            sr = g_model(lr)
            
            # Clone SR for discriminator to avoid CUDA graph issues
            sr_for_disc = sr.detach().clone()
            torch.cuda.synchronize()  # Ensure clone completes
            
            # Mark CUDA Graph step boundary if available
            if has_cuda_graph_marking:
                torch.compiler.cudagraph_mark_step_begin()
            
            # Use clone for discriminator input
            gt_clone = gt.detach().clone()
            torch.cuda.synchronize()
            
            if has_cuda_graph_marking:
                torch.compiler.cudagraph_mark_step_begin()
                
            gt_output = d_model(gt_clone)
            
            if has_cuda_graph_marking:
                torch.compiler.cudagraph_mark_step_begin()
                
            sr_output = d_model(sr)
            
            # Calculate losses
            pixel_loss = pixel_criterion(sr, gt)
            content_loss = feature_criterion(sr, gt)
            
            # Relativistic GAN loss
            d_loss_gt = adversarial_criterion(gt_output - torch.mean(sr_output), fake_label) * 0.5
            d_loss_sr = adversarial_criterion(sr_output - torch.mean(gt_output), real_label) * 0.5
            adversarial_loss = d_loss_gt + d_loss_sr
            
            # Apply loss weights
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))
            content_loss = torch.sum(torch.mul(content_weight, content_loss))
            adversarial_loss = torch.sum(torch.mul(adversarial_weight, adversarial_loss))
            
            # Total generator loss
            g_loss = pixel_loss + content_loss + adversarial_loss
        
        # Backpropagation with gradient scaling
        scaler.scale(g_loss).backward()
        scaler.step(g_optimizer)
        scaler.update()
        
        #--------------------------------
        # Train Discriminator
        #--------------------------------
        # Enable discriminator gradients
        for d_param in d_model.parameters():
            d_param.requires_grad = True
        
        # Zero discriminator gradients
        d_optimizer.zero_grad(set_to_none=True)
        
        # Consolidated discriminator training to reduce CUDA graph issues
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Mark CUDA Graph step boundary if available
            if has_cuda_graph_marking:
                torch.compiler.cudagraph_mark_step_begin()
                
            # Process real samples
            gt_output = d_model(gt)
            
            if has_cuda_graph_marking:
                torch.compiler.cudagraph_mark_step_begin()
                
            # Process fake samples - use the previously cloned tensor
            sr_output = d_model(sr_for_disc)
            
            # Calculate discriminator losses
            d_loss_gt = adversarial_criterion(gt_output - torch.mean(sr_output), real_label) * 0.5
            d_loss_sr = adversarial_criterion(sr_output - torch.mean(gt_output), fake_label) * 0.5
            
            # Combined discriminator loss
            d_loss = d_loss_gt + d_loss_sr
        
        # Single backward pass for discriminator
        scaler.scale(d_loss).backward()
        scaler.step(d_optimizer)
        scaler.update()
        
        # Update EMA model if enabled
        if config.model.ema.enable and ema_g_model is not None:
            ema_g_model.update_parameters(g_model)
        
        # Update metrics
        g_losses.update(g_loss.item(), batch_size)
        d_losses.update(d_loss.item(), batch_size)
        pixel_losses.update(pixel_loss.item(), batch_size)
        content_losses.update(content_loss.item(), batch_size)
        adversarial_losses.update(adversarial_loss.item(), batch_size)
        
        # Update batch time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log progress
        if batch_index % config.train.print_freq == 0 and is_main_process:
            progress.display(batch_index)
            
            # Log to text file
            if logger is not None:
                u.log_training(
                    logger['txt_path'],
                    f"Epoch [{epoch+1}][{batch_index}/{batches}] - "
                    f"G_Loss: {g_loss.item():.6f}, D_Loss: {d_loss.item():.6f}, "
                    f"Pixel: {pixel_loss.item():.6f}, Content: {content_loss.item():.6f}, "
                    f"Adv: {adversarial_loss.item():.6f}"
                )
    
    # Return metrics dictionary
    return {
        "g_loss": g_losses.avg,
        "d_loss": d_losses.avg,
        "pixel_loss": pixel_losses.avg,
        "content_loss": content_losses.avg,
        "adversarial_loss": adversarial_losses.avg
    }