import collections.abc
import typing
from itertools import repeat

import torch
from torch import Tensor

_I = typing.Optional[int]
_D = typing.Optional[torch.dtype]


def _to_tuple(dim: int):
    """Convert the input to a tuple

    Args:
        dim (int): the dimension of the input
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, dim))

    return parse


def _check_tensor_shape(raw_tensor: Tensor, dst_tensor: Tensor):
    """Check if the dimensions of the two tensors are the same

    Args:
        raw_tensor (np.ndarray or Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (np.ndarray or Tensor): reference image tensor flow, RGB format, data range [0, 1]
    """
    # Check if the tensor scale is consistent
    assert raw_tensor.shape == dst_tensor.shape, \
        f"Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"


def _reshape_input_torch(tensor: Tensor) -> typing.Tuple[Tensor, _I, _I, int, int]:
    """Reshape the input tensor to 4-dim tensor

    Args:
        tensor (Tensor): shape (b, c, h, w) or (c, h, w) or (h, w)

    Returns:
        tensor (Tensor): shape (b*c, 1, h, w)
    """
    if tensor.dim() == 4:
        b, c, h, w = tensor.size()
    elif tensor.dim() == 3:
        c, h, w = tensor.size()
        b = None
    elif tensor.dim() == 2:
        h, w = tensor.size()
        b = c = None
    else:
        raise ValueError(f"{tensor.dim()}-dim Tensor is not supported!")

    tensor = tensor.view(-1, 1, h, w)

    return tensor, b, c, h, w


def _cast_input_torch(tensor: Tensor) -> typing.Tuple[Tensor, _D]:
    """Casts the input tensor to the correct data type and stores the original data type.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with the correct data type.
    """
    if tensor.dtype != torch.float32 or tensor.dtype != torch.float64:
        dtype = tensor.dtype
        tensor = tensor.float()
    else:
        dtype = None

    return tensor, dtype


def _cast_output_torch(tensor: Tensor, dtype: _D) -> Tensor:
    """Convert tensor back to original dtype if needed

    Args:
        tensor (Tensor): Input tensor
        dtype (_D): Target dtype

    Returns:
        Tensor: Tensor with the original dtype
    """
    if dtype is not None:
        if not dtype.is_floating_point:
            tensor = tensor.round()
        # To prevent over/underflow when converting types
        if dtype is torch.uint8:
            tensor = tensor.clamp(0, 255)

        tensor = tensor.to(dtype=dtype)

    return tensor


def _reshape_tensor_torch(tensor: Tensor, b, c):
    """Reshape tensor back to original shape

    Args:
        tensor (Tensor): Input tensor
        b: Batch size
        c: Channel count

    Returns:
        Tensor: Reshaped tensor
    """
    if b is not None:
        tensor = tensor.view(b, c, tensor.shape[-2], tensor.shape[-1])
    else:
        if c is not None:
            tensor = tensor.view(c, tensor.shape[-2], tensor.shape[-1])
        else:
            tensor = tensor.view(tensor.shape[-2], tensor.shape[-1])
    
    return tensor