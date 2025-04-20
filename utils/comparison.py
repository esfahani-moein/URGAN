import warnings
import typing
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F_torch

from .color_space import rgb_to_ycbcr_torch


def _mse_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        only_test_y_channel: bool,
        data_range: float = 1.0,
        eps: float = 1e-8,
) -> Tensor:
    """PyTorch implements the MSE (Mean Squared Error, mean square error) function

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 1]
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        data_range (float, optional): Maximum value range of images. Default: 1.0
        eps (float, optional): Deviation prevention denominator is 0. Default: 1e-8

    Returns:
        mse_metrics (Tensor): MSE metrics
    """
    # Convert RGB tensor data to YCbCr tensor, and only extract Y channel data
    if only_test_y_channel and raw_tensor.shape[1] == 3 and dst_tensor.shape[1] == 3:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, True)

    # Efficient calculation without intermediate allocations
    diff = (raw_tensor * data_range - dst_tensor * data_range)
    mse_metrics = torch.mean(diff.pow(2) + eps, dim=[1, 2, 3])

    return mse_metrics


def _psnr_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        only_test_y_channel: bool,
        data_range: float = 1.0,
        eps: float = 1e-8,
) -> Tensor:
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 1]
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        data_range (float, optional): Maximum value range of images. Default: 1.0
        eps (float, optional): Deviation prevention denominator is 0. Default: 1e-8

    Returns:
        psnr_metrics (Tensor): PSNR metrics
    """
    # Calculate MSE
    mse_metrics = _mse_torch(raw_tensor, dst_tensor, only_test_y_channel, data_range, eps)

    # Calculate PSNR
    psnr_metrics = 10 * torch.log10((data_range ** 2) / (mse_metrics + eps))

    return psnr_metrics


def _ssim_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        gaussian_kernel_window: Tensor,
        downsampling: bool = False,
        get_ssim_map: bool = False,
        get_cs_map: bool = False,
        get_weight: bool = False,
        only_test_y_channel: bool = True,
        data_range: float = 255.0,
) -> Tensor | tuple[Any, Any] | tuple[Any, Tensor] | Any:
    """PyTorch implements SSIM (Structural Similarity)

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 1]
        gaussian_kernel_window (Tensor): 2D Gaussian kernel
        downsampling (bool): Whether to perform downsampling. Default: False
        get_ssim_map (bool): Whether to return SSIM map. Default: False
        get_cs_map (bool): Whether to return CS map. Default: False
        get_weight (bool): Whether to return weight map. Default: False
        only_test_y_channel (bool): Whether to test only the Y channel of the image. Default: True
        data_range (float): Maximum value range of images. Default: 255.0

    Returns:
        ssim_metrics (Tensor or tuple): SSIM metrics or tuple of metrics
    """
    # If input is RGB format and only Y channel is tested, 
    # the input RGB format data is converted to YCbCr format data, 
    # and only Y channel data is extracted
    if only_test_y_channel and raw_tensor.shape[1] == 3 and dst_tensor.shape[1] == 3:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, only_use_y_channel=True)
    else:
        # Check the value range of input data
        if torch.max(raw_tensor) > 1.0 or torch.max(dst_tensor) > 1.0:
            warnings.warn(
                f"The input image data range is expected to be [0, 1], but got [{torch.min(raw_tensor)}, {torch.max(raw_tensor)}]. "
                f"It will be normalized to [0, 1].")
            raw_tensor = raw_tensor / 255.0
            dst_tensor = dst_tensor / 255.0

        # Convert input tensor data type to specified data type
        raw_tensor = raw_tensor * data_range
        raw_tensor = raw_tensor - raw_tensor.detach() + raw_tensor.round()
        dst_tensor = dst_tensor * data_range
        dst_tensor = dst_tensor - dst_tensor.detach() + dst_tensor.round()

    # Move kernel to the same device and dtype as input tensors
    gaussian_kernel_window = gaussian_kernel_window.to(raw_tensor.device, dtype=raw_tensor.dtype)

    # Define SSIM constants
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # If the image size is large enough, downsample
    downsampling_factor = max(1, round(min(raw_tensor.size()[-2:]) / 256))
    if (downsampling_factor > 1) and downsampling:
        raw_tensor = F_torch.avg_pool2d(raw_tensor, kernel_size=(downsampling_factor, downsampling_factor))
        dst_tensor = F_torch.avg_pool2d(dst_tensor, kernel_size=(downsampling_factor, downsampling_factor))

    # Calculate mean using convolution
    mu1 = F_torch.conv2d(
        raw_tensor,
        gaussian_kernel_window,
        stride=(1, 1),
        padding=(0, 0),
        groups=raw_tensor.shape[1]
    )
    mu2 = F_torch.conv2d(
        dst_tensor,
        gaussian_kernel_window,
        stride=(1, 1),
        padding=(0, 0),
        groups=dst_tensor.shape[1]
    )

    # Calculate squared means and cross-correlation
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Pre-compute squared tensors for efficiency
    raw_tensor_sq = raw_tensor * raw_tensor
    dst_tensor_sq = dst_tensor * dst_tensor
    raw_dst_tensor = raw_tensor * dst_tensor
    
    # Calculate variances and covariance
    sigma1_sq = F_torch.conv2d(
        raw_tensor_sq,
        gaussian_kernel_window,
        stride=(1, 1),
        padding=(0, 0),
        groups=raw_tensor.shape[1]
    ) - mu1_sq
    
    sigma2_sq = F_torch.conv2d(
        dst_tensor_sq,
        gaussian_kernel_window,
        stride=(1, 1),
        padding=(0, 0),
        groups=dst_tensor.shape[1]
    ) - mu2_sq
    
    sigma12 = F_torch.conv2d(
        raw_dst_tensor,
        gaussian_kernel_window,
        stride=(1, 1),
        padding=(0, 0),
        groups=raw_tensor.shape[1]
    ) - mu1_mu2

    # Calculate SSIM components
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    
    # Force SSIM output to be non-negative to avoid negative results
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)

    # Weight map for visualization
    if get_weight:
        weight_map = gaussian_kernel_window.squeeze(0).squeeze(0)
        return weight_map

    # Calculate metrics based on specified return values
    ssim_val = torch.mean(ssim_map, dim=[1, 2, 3])
    cs = torch.mean(cs_map, dim=[1, 2, 3])

    if get_ssim_map and get_cs_map:
        return ssim_val, cs, ssim_map, cs_map
    elif get_ssim_map:
        return ssim_val, cs, ssim_map
    elif get_cs_map:
        return ssim_val, cs, cs_map
    else:
        return ssim_val