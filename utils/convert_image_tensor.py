import numpy as np
import re
import torch
from torch import Tensor
from typing import Any

def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image



def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor

def natsorted(iterable):
    """
    Sort an iterable naturally (human-friendly sorting).
    
    This function mimics the basic functionality of natsort.natsorted.
    It sorts strings containing numbers in a way that feels natural to humans:
    e.g., ['file1', 'file2', 'file10'] instead of ['file1', 'file10', 'file2']
    
    Args:
        iterable: An iterable containing strings to be sorted naturally
        
    Returns:
        A sorted list of the items from the iterable
    """
    def natural_keys(text):
        """
        Helper function for natural sorting.
        Splits text into text and numeric parts for proper sorting.
        """
        def atoi(text):
            """Convert text to integer if it's a digit, otherwise keep as is."""
            return int(text) if text.isdigit() else text
            
        # Split by numbers (one or more digits)
        return [atoi(c) for c in re.split(r'(\d+)', text)]
    
    # Convert the iterable to a list and sort it using the natural keys function
    return sorted(list(iterable), key=natural_keys)