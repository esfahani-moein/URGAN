"""
Color space conversion utilities optimized for image processing tasks.
Provides both NumPy and PyTorch implementations with consistent output.
"""

import numpy as np
import torch
from torch import Tensor
import cv2
from typing import Union, Tuple, Optional


def rgb_to_ycbcr(image: np.ndarray, only_use_y_channel: bool = False) -> np.ndarray:
    """Convert RGB image to YCbCr color space.
    
    Args:
        image: RGB image with shape (H, W, 3) and range [0, 1]
        only_use_y_channel: If True, return only Y channel
        
    Returns:
        YCbCr image with same shape as input (or single channel if only_use_y_channel=True)
    """
    # Check if OpenCV can be used directly
    if image.dtype == np.uint8:
        # OpenCV expects uint8 range [0, 255]
        result = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        # OpenCV returns YCrCb, not YCbCr, so we need to swap the channels
        if not only_use_y_channel:
            result = result[..., (0, 2, 1)]
        else:
            result = result[..., 0]
        return result.astype(np.float32) / 255.0
    
    # For float inputs, use the ITU-R BT.601 conversion formulas
    if only_use_y_channel:
        return np.dot(image, [65.481, 128.553, 24.966]) / 255.0 + 16.0 / 255.0
    else:
        matrix = np.array([
            [65.481, 128.553, 24.966],
            [-37.797, -74.203, 112.0],
            [112.0, -93.786, -18.214]
        ]) / 255.0
        bias = np.array([16, 128, 128]) / 255.0
        
        result = np.dot(image, matrix.T) + bias
        return result.astype(np.float32)


def bgr_to_ycbcr(image: np.ndarray, only_use_y_channel: bool = False) -> np.ndarray:
    """Convert BGR image to YCbCr color space.
    
    Args:
        image: BGR image with shape (H, W, 3) and range [0, 1]
        only_use_y_channel: If True, return only Y channel
        
    Returns:
        YCbCr image with same shape as input (or single channel if only_use_y_channel=True)
    """
    # Check if OpenCV can be used directly
    if image.dtype == np.uint8:
        # Convert BGR to RGB first, then to YCbCr
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_to_ycbcr(rgb_image, only_use_y_channel)
    
    # For float inputs, use the ITU-R BT.601 conversion formulas with swapped channels
    if only_use_y_channel:
        return np.dot(image, [24.966, 128.553, 65.481]) / 255.0 + 16.0 / 255.0
    else:
        matrix = np.array([
            [24.966, 128.553, 65.481],
            [112.0, -74.203, -37.797],
            [-18.214, -93.786, 112.0]
        ]) / 255.0
        bias = np.array([16, 128, 128]) / 255.0
        
        result = np.dot(image, matrix.T) + bias
        return result.astype(np.float32)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert YCbCr image to RGB color space.
    
    Args:
        image: YCbCr image with shape (H, W, 3) and range [0, 1]
        
    Returns:
        RGB image with same shape as input
    """
    # Check if OpenCV can be used directly
    if image.dtype == np.uint8:
        # OpenCV expects YCrCb, not YCbCr
        image_ycrcb = image[..., (0, 2, 1)]
        return cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2RGB)
    
    # For float inputs, use the ITU-R BT.601 inverse conversion
    image = image.copy() * 255.0
    
    matrix = np.array([
        [0.00456621, 0.00456621, 0.00456621],
        [0, -0.00153632, 0.00791071],
        [0.00625893, -0.00318811, 0]
    ]) * 255.0
    
    bias = np.array([-222.921, 135.576, -276.836]) / 255.0
    
    result = np.dot(image, matrix) + bias
    return np.clip(result, 0, 1).astype(np.float32)


def ycbcr_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert YCbCr image to BGR color space.
    
    Args:
        image: YCbCr image with shape (H, W, 3) and range [0, 1]
        
    Returns:
        BGR image with same shape as input
    """
    # Convert to RGB first, then to BGR
    rgb_image = ycbcr_to_rgb(image)
    
    if image.dtype == np.uint8:
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # For float inputs, swap the channels
    return rgb_image[..., ::-1]


def expand_y(image: np.ndarray) -> np.ndarray:
    """Convert image to YCbCr and extract Y channel with shape (H, W, 1).
    
    Args:
        image: BGR image with shape (H, W, 3) and range [0, 255]
        
    Returns:
        Y channel image with shape (H, W, 1) and range [0, 255]
    """
    # Normalize image to [0, 1]
    image_float = image.astype(np.float32) / 255.0
    
    # Convert BGR to YCbCr and get Y channel
    y_channel = bgr_to_ycbcr(image_float, only_use_y_channel=True)
    
    # Expand dimensions and restore range to [0, 255]
    y_channel = np.expand_dims(y_channel, axis=2) * 255.0
    
    return y_channel


# PyTorch implementations

def rgb_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool = False) -> Tensor:
    """Convert RGB tensor to YCbCr color space.
    
    Args:
        tensor: RGB tensor with shape (B, 3, H, W) and range [0, 1]
        only_use_y_channel: If True, return only Y channel
        
    Returns:
        YCbCr tensor with shape (B, 3, H, W) or (B, 1, H, W) if only_use_y_channel=True
    """
    if only_use_y_channel:
        weight = torch.tensor([[65.481], [128.553], [24.966]], 
                              dtype=tensor.dtype, device=tensor.device) / 255.0
        bias = torch.tensor([16.0/255.0], dtype=tensor.dtype, device=tensor.device)
        
        # Efficient implementation without permuting tensors
        tensor = torch.sum(tensor * weight.view(1, 3, 1, 1), dim=1, keepdim=True) + bias
        return tensor
    else:
        weight = torch.tensor([
            [65.481, -37.797, 112.0],
            [128.553, -74.203, -93.786],
            [24.966, 112.0, -18.214]
        ], dtype=tensor.dtype, device=tensor.device) / 255.0
        
        bias = torch.tensor([16.0, 128.0, 128.0], 
                           dtype=tensor.dtype, 
                           device=tensor.device).view(1, 3, 1, 1) / 255.0
        
        # Use einsum for efficient matrix multiplication
        tensor = torch.einsum('bchw,cd->bdhw', tensor, weight).contiguous()
        # Reshape to match original tensor dimensions
        tensor = tensor.view_as(tensor) + bias
        
        return tensor


def bgr_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool = False) -> Tensor:
    """Convert BGR tensor to YCbCr color space.
    
    Args:
        tensor: BGR tensor with shape (B, 3, H, W) and range [0, 1]
        only_use_y_channel: If True, return only Y channel
        
    Returns:
        YCbCr tensor with shape (B, 3, H, W) or (B, 1, H, W) if only_use_y_channel=True
    """
    if only_use_y_channel:
        weight = torch.tensor([[24.966], [128.553], [65.481]], 
                              dtype=tensor.dtype, device=tensor.device) / 255.0
        bias = torch.tensor([16.0/255.0], dtype=tensor.dtype, device=tensor.device)
        
        # Efficient implementation without permuting tensors
        tensor = torch.sum(tensor * weight.view(1, 3, 1, 1), dim=1, keepdim=True) + bias
        return tensor
    else:
        weight = torch.tensor([
            [24.966, 112.0, -18.214],
            [128.553, -74.203, -93.786],
            [65.481, -37.797, 112.0]
        ], dtype=tensor.dtype, device=tensor.device) / 255.0
        
        bias = torch.tensor([16.0, 128.0, 128.0], 
                           dtype=tensor.dtype, 
                           device=tensor.device).view(1, 3, 1, 1) / 255.0
        
        # Use einsum for efficient matrix multiplication
        tensor = torch.einsum('bchw,cd->bdhw', tensor, weight).contiguous()
        # Reshape to match original tensor dimensions
        tensor = tensor.view_as(tensor) + bias
        
        return tensor


def ycbcr_to_rgb_torch(tensor: Tensor) -> Tensor:
    """Convert YCbCr tensor to RGB color space.
    
    Args:
        tensor: YCbCr tensor with shape (B, 3, H, W) and range [0, 1]
        
    Returns:
        RGB tensor with shape (B, 3, H, W) and range [0, 1]
    """
    weight = torch.tensor([
        [0.00456621, 0, 0.00625893],
        [0.00456621, -0.00153632, -0.00318811],
        [0.00456621, 0.00791071, 0]
    ], dtype=tensor.dtype, device=tensor.device) * 255.0
    
    bias = torch.tensor([-222.921, 135.576, -276.836], 
                       dtype=tensor.dtype, 
                       device=tensor.device).view(1, 3, 1, 1) / 255.0
    
    # Scale input tensor
    tensor = tensor * 255.0
    
    # Apply transformation
    tensor = torch.einsum('bchw,cd->bdhw', tensor, weight).contiguous()
    tensor = tensor + bias
    
    # Rescale back to [0, 1]
    tensor = tensor / 255.0
    
    # Clamp values to valid range
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    return tensor


def ycbcr_to_bgr_torch(tensor: Tensor) -> Tensor:
    """Convert YCbCr tensor to BGR color space.
    
    Args:
        tensor: YCbCr tensor with shape (B, 3, H, W) and range [0, 1]
        
    Returns:
        BGR tensor with shape (B, 3, H, W) and range [0, 1]
    """
    weight = torch.tensor([
        [0.00456621, 0, 0.00625893],  # R
        [0.00456621, -0.00153632, -0.00318811],  # G
        [0.00456621, 0.00791071, 0]  # B
    ], dtype=tensor.dtype, device=tensor.device) * 255.0
    
    # Flip the weight matrix for BGR
    weight = weight.flip(0)
    
    bias = torch.tensor([-276.836, 135.576, -222.921], 
                       dtype=tensor.dtype, 
                       device=tensor.device).view(1, 3, 1, 1) / 255.0
    
    # Scale input tensor
    tensor = tensor * 255.0
    
    # Apply transformation
    tensor = torch.einsum('bchw,cd->bdhw', tensor, weight).contiguous()
    tensor = tensor + bias
    
    # Rescale back to [0, 1]
    tensor = tensor / 255.0
    
    # Clamp values to valid range
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    return tensor


# Unified conversion interface with library acceleration when possible

def convert_colorspace(
    image: Union[np.ndarray, Tensor], 
    source: str, 
    target: str,
    only_y_channel: bool = False
) -> Union[np.ndarray, Tensor]:
    """Universal color space conversion function that uses optimized library implementations when possible.
    
    Args:
        image: Input image as numpy array or PyTorch tensor
        source: Source color space ('rgb', 'bgr', or 'ycbcr')
        target: Target color space ('rgb', 'bgr', 'ycbcr', or 'y')
        only_y_channel: Whether to return only Y channel when target is 'ycbcr'
        
    Returns:
        Converted image in the target color space
    """
    source = source.lower()
    target = target.lower()
    
    # Handle case where source and target are the same
    if source == target and not (target == 'ycbcr' and only_y_channel):
        return image
    
    # PyTorch tensor implementation
    if torch.is_tensor(image):
        if source == 'rgb' and target == 'ycbcr':
            return rgb_to_ycbcr_torch(image, only_y_channel)
        elif source == 'bgr' and target == 'ycbcr':
            return bgr_to_ycbcr_torch(image, only_y_channel)
        elif source == 'ycbcr' and target == 'rgb':
            return ycbcr_to_rgb_torch(image)
        elif source == 'ycbcr' and target == 'bgr':
            return ycbcr_to_bgr_torch(image)
        elif source == 'rgb' and target == 'bgr':
            # Simply swap the channels
            return image.flip(1)
        elif source == 'bgr' and target == 'rgb':
            # Simply swap the channels
            return image.flip(1)
        elif target == 'y':
            # Convert to YCbCr first, then extract Y
            if source == 'rgb':
                return rgb_to_ycbcr_torch(image, True)
            elif source == 'bgr':
                return bgr_to_ycbcr_torch(image, True)
        else:
            raise ValueError(f"Unsupported conversion: {source} to {target}")
    
    # NumPy array implementation
    else:
        # Use OpenCV for uint8 images when possible (much faster)
        if image.dtype == np.uint8:
            cv_conversions = {
                ('rgb', 'bgr'): cv2.COLOR_RGB2BGR,
                ('bgr', 'rgb'): cv2.COLOR_BGR2RGB,
                ('rgb', 'ycbcr'): cv2.COLOR_RGB2YCrCb,  # Note: OpenCV uses YCrCb, not YCbCr
                ('bgr', 'ycbcr'): cv2.COLOR_BGR2YCrCb,
                ('ycbcr', 'rgb'): cv2.COLOR_YCrCb2RGB,
                ('ycbcr', 'bgr'): cv2.COLOR_YCrCb2BGR,
            }
            
            key = (source, 'ycbcr' if target == 'y' else target)
            
            if key in cv_conversions:
                result = cv2.cvtColor(image, cv_conversions[key])
                
                # Handle YCrCb to YCbCr conversion (swap Cr and Cb channels)
                if target == 'ycbcr' and not only_y_channel:
                    result = result[..., (0, 2, 1)]
                
                # Extract only Y channel if needed
                if target == 'y' or only_y_channel:
                    result = result[..., 0:1] if target == 'y' else result[..., 0]
                
                return result
        
        # Fall back to custom implementations for float arrays
        if source == 'rgb' and target == 'ycbcr':
            return rgb_to_ycbcr(image, only_y_channel)
        elif source == 'bgr' and target == 'ycbcr':
            return bgr_to_ycbcr(image, only_y_channel)
        elif source == 'ycbcr' and target == 'rgb':
            return ycbcr_to_rgb(image)
        elif source == 'ycbcr' and target == 'bgr':
            return ycbcr_to_bgr(image)
        elif source == 'rgb' and target == 'bgr':
            return image[..., ::-1]
        elif source == 'bgr' and target == 'rgb':
            return image[..., ::-1]
        elif target == 'y':
            # Convert to YCbCr first, then extract Y
            if source == 'rgb':
                return rgb_to_ycbcr(image, True)
            elif source == 'bgr':
                return bgr_to_ycbcr(image, True)
            elif source == 'ycbcr':
                return image[..., 0]
        else:
            raise ValueError(f"Unsupported conversion: {source} to {target}")