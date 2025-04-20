import math
import typing

import torch
from torch import Tensor
from torch.nn import functional as F_torch

from .kernel_handling import _get_weight_torch
from .tensor_handling import (_cast_input_torch, _cast_output_torch, 
                          _reshape_input_torch, _reshape_tensor_torch)


def _reflect_padding_torch(tensor: Tensor, dim: int, pad_pre: int, pad_post: int) -> Tensor:
    """Reflect padding for 2-dim tensor

    Args:
        tensor (Tensor): shape (b, c, h, w)
        dim (int): 2 or -2
        pad_pre (int): padding size before the dim
        pad_post (int): padding size after the dim

    Returns:
        padding_buffer (Tensor): shape (b, c, h + pad_pre + pad_post, w) or (b, c, h, w + pad_pre + pad_post)
    """
    b, c, h, w = tensor.size()

    if dim == 2 or dim == -2:
        padding_buffer = tensor.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(tensor)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(tensor[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(tensor[..., -(p + 1), :])
    else:
        padding_buffer = tensor.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(tensor)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(tensor[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(tensor[..., -(p + 1)])

    return padding_buffer


def _padding_torch(
        tensor: Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int,
        padding_type: typing.Optional[str] = "reflect",
) -> Tensor:
    """Apply padding to tensor"""
    if padding_type is None:
        return tensor
    elif padding_type == "reflect":
        x_pad = _reflect_padding_torch(tensor, dim, pad_pre, pad_post)
    else:
        raise ValueError(f"{padding_type} padding is not supported!")

    return x_pad


def _get_padding_torch(tensor: Tensor, kernel_size: int, x_size: int) -> typing.Tuple[int, int, Tensor]:
    """Get padding size and padded tensor

    Args:
        tensor (Tensor): shape (b, c, h, w)
        kernel_size (int): kernel size
        x_size (int): input size

    Returns:
        pad_pre (int): padding size before the dim
    """
    tensor = tensor.long()
    r_min = tensor.min()
    r_max = tensor.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        tensor += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, tensor


def _reshape_tensor_torch(tensor: Tensor, dim: int, kernel_size: int) -> Tensor:
    """Reshape the tensor to the shape of (B, C, K, H, W) or (B, C, H, K, W) for 1D convolution.

    Args:
        tensor (Tensor): Tensor to be reshaped.
        dim (int): Dimension to be resized.
        kernel_size (int): Size of the kernel.

    Returns:
        Tensor: Reshaped tensor.
    """
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = tensor.size(-2) - kernel_size + 1
        w_out = tensor.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = tensor.size(-2)
        w_out = tensor.size(-1) - kernel_size + 1

    unfold = F_torch.unfold(tensor, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)

    return unfold


def _resize_1d_torch(
        tensor: Tensor,
        dim: int,
        size: int,
        scale: float,
        kernel: str = "cubic",
        sigma: float = 2.0,
        padding_type: str = "reflect",
        antialiasing: bool = True,
) -> Tensor:
    """Resize the given tensor to the given size.

    Args:
        tensor (Tensor): Tensor to be resized.
        dim (int): Dimension to be resized.
        size (int): Size of the resized dimension.
        scale (float): Scale factor of the resized dimension.
        kernel (str, optional): Kernel type. Default: ``cubic``
        sigma (float, optional): Sigma of the gaussian kernel. Default: 2.0
        padding_type (str, optional): Padding type. Default: ``reflect``
        antialiasing (bool, optional): Whether to use antialiasing. Default: ``True``

    Returns:
        Tensor: Resized tensor.
    """
    # Identity case
    if scale == 1:
        return tensor

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel == "cubic":
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sizes
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        pos = torch.linspace(
            0,
            size - 1,
            steps=size,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        pos = (pos + 0.5) / scale - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = _get_weight_torch(
            dist,
            kernel_size,
            kernel,
            sigma,
            antialiasing_factor,
        )
        pad_pre, pad_post, base = _get_padding_torch(base, kernel_size, tensor.size(dim))

    # To back-propagate through x
    x_pad = _padding_torch(tensor, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = _reshape_tensor_torch(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    tensor = sample * weight
    tensor = tensor.sum(dim=1, keepdim=True)

    return tensor


def _downsampling_2d_torch(
        tensor: Tensor,
        k: Tensor,
        scale: int,
        padding_type: str = "reflect",
) -> Tensor:
    """Apply 2D downsampling with a kernel

    Args:
        tensor (Tensor): Input tensor
        k (Tensor): Kernel
        scale (int): Downsampling scale
        padding_type (str): Type of padding

    Returns:
        Tensor: Downsampled tensor
    """
    c = tensor.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)

    k = k.to(dtype=tensor.dtype, device=tensor.device)
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e

    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    tensor = _padding_torch(tensor, -2, pad_h, pad_h, padding_type=padding_type)
    tensor = _padding_torch(tensor, -1, pad_w, pad_w, padding_type=padding_type)
    y = F_torch.conv2d(tensor, k, padding=0, stride=scale)
    return y


def _image_resize_torch(
        x: Tensor,
        scale_factor: typing.Optional[float] = None,
        sizes: typing.Optional[typing.Tuple[int, int]] = None,
        kernel: typing.Union[str, Tensor] = "cubic",
        sigma: float = 2,
        padding_type: str = "reflect",
        antialiasing: bool = True,
) -> Tensor:
    """Resize image with given kernel and sigma.

    Args:
        x (Tensor): Input image with shape (b, c, h, w)
        scale_factor (float): Scale factor for resizing
        sizes (tuple): Size of the output image (h, w)
        kernel (str or Tensor, optional): Kernel type or kernel tensor. Default: ``cubic``
        sigma (float): Sigma for Gaussian kernel. Default: 2
        padding_type (str): Padding type for convolution. Default: ``reflect``
        antialiasing (bool): Whether to use antialiasing or not. Default: ``True``

    Returns:
        Tensor: Resized image with shape (b, c, h, w)
    """
    # Only one zoom factor and target size can be selected
    if scale_factor is None and sizes is None:
        raise ValueError("One of scale or sizes must be specified!")
    if scale_factor is not None and sizes is not None:
        raise ValueError("Please specify scale or sizes to avoid conflict!")

    # Reshape the input tensor to 4-dim tensor
    x, b, c, h, w = _reshape_input_torch(x)

    scales = (1.0, 1.0)

    # Determine output size
    if sizes is None and scale_factor is not None:
        sizes = (math.ceil(h * scale_factor), math.ceil(w * scale_factor))
        scales = (scale_factor, scale_factor)

    # Determine output scale
    if scale_factor is None and sizes is not None:
        scales = (sizes[0] / h, sizes[1] / w)

    # Casts the input tensor to the correct data type and stores the original data type.
    x, dtype = _cast_input_torch(x)

    if isinstance(kernel, str) and sizes is not None:
        # Core resizing module
        x = _resize_1d_torch(
            x,
            -2,
            sizes[0],
            scales[0],
            kernel,
            sigma,
            padding_type,
            antialiasing)
        x = _resize_1d_torch(
            x,
            -1,
            sizes[1],
            scales[1],
            kernel,
            sigma,
            padding_type,
            antialiasing)
    elif isinstance(kernel, torch.Tensor) and scale_factor is not None:
        x = _downsampling_2d_torch(x, kernel, scale=int(1 / scale_factor))

    x = _reshape_tensor_torch(x, b, c)
    x = _cast_output_torch(x, dtype)
    return x