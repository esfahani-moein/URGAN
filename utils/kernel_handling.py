import numpy as np
import torch
from torch import Tensor

from .tensor_handling import _to_tuple


def _fspecial_gaussian_torch(
        window_size: int,
        sigma: float,
        channels: int = 3,
        filter_type: int = 0,
) -> Tensor:
    """PyTorch implements the fspecial_gaussian() function in MATLAB

    Args:
        window_size (int): Gaussian filter size
        sigma (float): sigma parameter in Gaussian filter
        channels (int): number of image channels, default: ``3``
        filter_type (int): filter type, 0: Gaussian filter, 1: mean filter, default: ``0``

    Returns:
        gaussian_kernel_window (Tensor): Gaussian filter
    """
    # Gaussian filter processing
    if filter_type == 0:
        shape = _to_tuple(2)(window_size)
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        g = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        g[g < np.finfo(g.dtype).eps * g.max()] = 0
        sum_height = g.sum()

        if sum_height != 0:
            g /= sum_height

        g = torch.from_numpy(g).float().repeat(channels, 1, 1, 1)

        return g
    # mean filter processing
    elif filter_type == 1:
        raise NotImplementedError(f"Only support `gaussian filter`, got {filter_type}")


def _cubic_contribution_torch(tensor: Tensor, a: float = -0.5) -> Tensor:
    """Calculate cubic interpolation weights"""
    ax = tensor.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=tensor.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=tensor.dtype)

    cont = cont_01 + cont_12
    return cont


def _gaussian_contribution_torch(x: Tensor, sigma: float = 2.0) -> Tensor:
    """Calculate Gaussian interpolation weights"""
    range_3sigma = (x.abs() <= 3 * sigma + 1)
    # Normalization will be done after
    cont = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    cont = cont * range_3sigma.to(dtype=x.dtype)
    return cont


def _get_weight_torch(
        tensor: Tensor,
        kernel_size: int,
        kernel: str = "cubic",
        sigma: float = 2.0,
        antialiasing_factor: float = 1,
) -> Tensor:
    """Get weight for each pixel

    Args:
        tensor (Tensor): shape (b, c, h, w)
        kernel_size (int): kernel size
        kernel (str): kernel type, cubic or gaussian
        sigma (float): sigma for gaussian kernel
        antialiasing_factor (float): antialiasing factor

    Returns:
        weight (Tensor): shape (b, c, k, h, w) or (b, c, h, k, w)
    """
    buffer_pos = tensor.new_zeros(kernel_size, len(tensor))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(tensor - idx)

    # Expand (downsampling) / Shrink (upsampling) the receptive field.
    buffer_pos *= antialiasing_factor
    if kernel == "cubic":
        weight = _cubic_contribution_torch(buffer_pos)
    elif kernel == "gaussian":
        weight = _gaussian_contribution_torch(buffer_pos, sigma=sigma)
    else:
        raise ValueError(f"{kernel} kernel is not supported!")

    weight /= weight.sum(dim=0, keepdim=True)
    return weight