"""Data augmentation utilities for super-resolution models."""

import random
import cv2
import torch
from typing import Union, List, Tuple, Optional, Any
import torchvision.transforms.functional as F_vision
import numpy as np

# Type aliases for cleaner annotations
Tensor = torch.Tensor
ndarray = np.ndarray
ImageType = Union[ndarray, Tensor, List[ndarray], List[Tensor]]

def random_crop_torch(
        gt_images: ImageType,
        lr_images: ImageType,
        gt_patch_size: int,
        upscale_factor: int,
) -> Tuple[Union[ndarray, Tensor, List[ndarray], List[Tensor]], 
           Union[ndarray, Tensor, List[ndarray], List[Tensor]]]:
    """Randomly crop patches from image pairs.

    Args:
        gt_images: Ground truth high-resolution images
        lr_images: Low resolution images
        gt_patch_size: Size of ground truth patches
        upscale_factor: Scaling factor between LR and HR images

    Returns:
        Tuple of (cropped_gt_images, cropped_lr_images)
    """
    # Convert single images to lists for consistent processing
    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # Detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    # Get dimensions based on image type
    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Calculate the size of the low-resolution patches
    lr_patch_size = gt_patch_size // upscale_factor

    # Find random crop coordinates
    lr_top = random.randint(0, lr_image_height - lr_patch_size)
    lr_left = random.randint(0, lr_image_width - lr_patch_size)

    # Crop low-resolution images
    if input_type == "Tensor":
        lr_images = [lr_image[
                     :,
                     :,
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size] for lr_image in lr_images]
    else:
        lr_images = [lr_image[
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size,
                     ...] for lr_image in lr_images]

    # Calculate HR crop coordinates and crop the ground truth images
    gt_top, gt_left = int(lr_top * upscale_factor), int(lr_left * upscale_factor)

    if input_type == "Tensor":
        gt_images = [v[
                     :,
                     :,
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size] for v in gt_images]
    else:
        gt_images = [v[
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size,
                     ...] for v in gt_images]

    # Return single images if only one was provided
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_rotate_torch(
        gt_images: ImageType,
        lr_images: ImageType,
        upscale_factor: int,
        angles: List[int],
        gt_center: Optional[Tuple[int, int]] = None,
        lr_center: Optional[Tuple[int, int]] = None,
        rotate_scale_factor: float = 1.0
) -> Tuple[Union[ndarray, Tensor, List[ndarray], List[Tensor]], 
           Union[ndarray, Tensor, List[ndarray], List[Tensor]]]:
    """Randomly rotate image pairs.

    Args:
        gt_images: Ground truth high-resolution images
        lr_images: Low resolution images
        upscale_factor: Scaling factor between LR and HR images
        angles: List of possible rotation angles in degrees
        gt_center: Center point for HR rotation (optional)
        lr_center: Center point for LR rotation (optional)
        rotate_scale_factor: Scale factor for rotation

    Returns:
        Tuple of (rotated_gt_images, rotated_lr_images)
    """
    # Randomly choose rotation angle
    angle = random.choice(angles)

    # Convert single images to lists for consistent processing
    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # Detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    # Get dimensions based on image type
    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Define LR rotation center if not provided
    if lr_center is None:
        lr_center = [lr_image_width // 2, lr_image_height // 2]

    # Rotate LR images based on type
    if input_type == "Tensor":
        # Use torchvision's functional rotate which is optimized for tensors
        lr_images = [F_vision.rotate(lr_image, angle, center=lr_center) for lr_image in lr_images]
    else:
        # Use OpenCV for numpy arrays
        lr_matrix = cv2.getRotationMatrix2D(lr_center, angle, rotate_scale_factor)
        lr_images = [cv2.warpAffine(lr_image, lr_matrix, (lr_image_width, lr_image_height)) 
                     for lr_image in lr_images]

    # Calculate HR dimensions and center
    gt_image_width = int(lr_image_width * upscale_factor)
    gt_image_height = int(lr_image_height * upscale_factor)

    if gt_center is None:
        gt_center = [gt_image_width // 2, gt_image_height // 2]

    # Rotate HR images based on type
    if input_type == "Tensor":
        gt_images = [F_vision.rotate(gt_image, angle, center=gt_center) for gt_image in gt_images]
    else:
        gt_matrix = cv2.getRotationMatrix2D(gt_center, angle, rotate_scale_factor)
        gt_images = [cv2.warpAffine(gt_image, gt_matrix, (gt_image_width, gt_image_height)) 
                     for gt_image in gt_images]

    # Return single images if only one was provided
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_horizontally_flip_torch(
        gt_images: ImageType,
        lr_images: ImageType,
        p: float = 0.5
) -> Tuple[Union[ndarray, Tensor, List[ndarray], List[Tensor]], 
           Union[ndarray, Tensor, List[ndarray], List[Tensor]]]:
    """Randomly flip images horizontally with probability p.

    Args:
        gt_images: Ground truth high-resolution images
        lr_images: Low resolution images
        p: Probability of flipping (default: 0.5)

    Returns:
        Tuple of (flipped_gt_images, flipped_lr_images)
    """
    # Determine if flip should be applied
    flip_prob = random.random()

    # Convert single images to lists for consistent processing
    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # Detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    # Apply flip if probability threshold is exceeded
    if flip_prob > p:
        if input_type == "Tensor":
            lr_images = [F_vision.hflip(lr_image) for lr_image in lr_images]
            gt_images = [F_vision.hflip(gt_image) for gt_image in gt_images]
        else:
            lr_images = [cv2.flip(lr_image, 1) for lr_image in lr_images]
            gt_images = [cv2.flip(gt_image, 1) for gt_image in gt_images]

    # Return single images if only one was provided
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_vertically_flip_torch(
        gt_images: ImageType,
        lr_images: ImageType,
        p: float = 0.5
) -> Tuple[Union[ndarray, Tensor, List[ndarray], List[Tensor]], 
           Union[ndarray, Tensor, List[ndarray], List[Tensor]]]:
    """Randomly flip images vertically with probability p.

    Args:
        gt_images: Ground truth high-resolution images
        lr_images: Low resolution images
        p: Probability of flipping (default: 0.5)

    Returns:
        Tuple of (flipped_gt_images, flipped_lr_images)
    """
    # Determine if flip should be applied
    flip_prob = random.random()

    # Convert single images to lists for consistent processing
    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # Detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    # Apply flip if probability threshold is exceeded
    if flip_prob > p:
        if input_type == "Tensor":
            lr_images = [F_vision.vflip(lr_image) for lr_image in lr_images]
            gt_images = [F_vision.vflip(gt_image) for gt_image in gt_images]
        else:
            lr_images = [cv2.flip(lr_image, 0) for lr_image in lr_images]
            gt_images = [cv2.flip(gt_image, 0) for gt_image in gt_images]

    # Return single images if only one was provided
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def apply_augmentations(
        gt: Union[ndarray, Tensor],
        lr: Union[ndarray, Tensor],
        config: Any
) -> Tuple[Union[ndarray, Tensor], Union[ndarray, Tensor]]:
    """Apply a series of data augmentations to image pairs.
    
    Args:
        gt: Ground truth high-resolution image
        lr: Low-resolution input image
        config: Configuration with augmentation parameters
        
    Returns:
        Tuple of (augmented_gt, augmented_lr)
    """
    # Apply random crop
    gt, lr = random_crop_torch(
        gt, lr, 
        config.train.dataset.gt_image_size, 
        config.scale
    )
    
    # Apply random rotations
    gt, lr = random_rotate_torch(
        gt, lr, 
        config.scale, 
        [0, 90, 180, 270]
    )
    
    # Apply random flips
    gt, lr = random_vertically_flip_torch(gt, lr)
    gt, lr = random_horizontally_flip_torch(gt, lr)
    
    return gt, lr