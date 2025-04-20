import torch

def model_test(model, test_dataloader, psnr_model, ssim_model, device):
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
