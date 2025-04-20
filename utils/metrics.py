import torch
from torch import nn, Tensor
import torch.backends.mps
from torch import distributed as dist
from enum import Enum
from typing import List, Any, Tuple

from .comparison import _mse_torch, _psnr_torch, _ssim_torch
from .kernel_handling import _fspecial_gaussian_torch
from .tensor_handling import _check_tensor_shape



class MSE(nn.Module):
    """PyTorch implements the MSE (Mean Squared Error, mean square error) function"""

    def __init__(self, crop_border: int = 0, only_test_y_channel: bool = True, **kwargs) -> None:
        """
        Args:
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            mse_metrics (Tensor): MSE metrics
        """
        super(MSE, self).__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.kwargs = kwargs

    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        # Check if two tensor scales are similar
        _check_tensor_shape(raw_tensor, dst_tensor)

        # crop pixel boundaries
        if self.crop_border > 0:
            raw_tensor = raw_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            dst_tensor = dst_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        mse_metrics = _mse_torch(raw_tensor, dst_tensor, self.only_test_y_channel, **self.kwargs)

        return mse_metrics
    

class PSNR(nn.Module):
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function"""

    def __init__(self, crop_border: int = 0, only_test_y_channel: bool = True, **kwargs) -> None:
        """
        Args:
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            psnr_metrics (Tensor): PSNR metrics
        """
        super(PSNR, self).__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.kwargs = kwargs

    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        # Check if two tensor scales are similar
        _check_tensor_shape(raw_tensor, dst_tensor)

        # crop pixel boundaries
        if self.crop_border > 0:
            raw_tensor = raw_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            dst_tensor = dst_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        psnr_metrics = _psnr_torch(raw_tensor, dst_tensor, self.only_test_y_channel, **self.kwargs)

        return psnr_metrics
    

class SSIM(nn.Module):
    """PyTorch implements SSIM (Structural Similarity) function"""

    def __init__(
            self,
            window_size: int = 11,
            gaussian_sigma: float = 1.5,
            channels: int = 3,
            downsampling: bool = False,
            get_ssim_map: bool = False,
            get_cs_map: bool = False,
            get_weight: bool = False,
            crop_border: int = 0,
            only_test_y_channel: bool = True,
            **kwargs,
    ) -> None:
        """
        Args:
            window_size (int): Gaussian filter size, must be an odd number, default: ``11``
            gaussian_sigma (float): sigma parameter in Gaussian filter, default: ``1.5``
            channels (int): number of image channels, default: ``3``
            downsampling (bool): Whether to perform downsampling, default: ``False``
            get_ssim_map (bool): Whether to return SSIM image, default: ``False``
            get_cs_map (bool): whether to return CS image, default: ``False``
            get_weight (bool): whether to return the weight image, default: ``False``
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            ssim_metrics (Tensor): SSIM metrics
        """
        super(SSIM, self).__init__()
        if only_test_y_channel and channels != 1:
            channels = 1
            
        # Register the Gaussian kernel as a buffer so it's properly moved to the right device
        self.register_buffer('gaussian_kernel_window', 
                            _fspecial_gaussian_torch(window_size, gaussian_sigma, channels))
        
        self.downsampling = downsampling
        self.get_ssim_map = get_ssim_map
        self.get_cs_map = get_cs_map
        self.get_weight = get_weight
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        
        # Default data range for SSIM
        kwargs.setdefault('data_range', 255.0)
        self.kwargs = kwargs

    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        # Check if two tensor scales are similar
        _check_tensor_shape(raw_tensor, dst_tensor)

        # crop pixel boundaries
        if self.crop_border > 0:
            raw_tensor = raw_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            dst_tensor = dst_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            
        # Use torch.no_grad for inference if not training
        if not self.training:
            with torch.no_grad():
                ssim_metrics = _ssim_torch(raw_tensor,
                                        dst_tensor,
                                        self.gaussian_kernel_window,
                                        self.downsampling,
                                        self.get_ssim_map,
                                        self.get_cs_map,
                                        self.get_weight,
                                        self.only_test_y_channel,
                                        **self.kwargs)
        else:
            ssim_metrics = _ssim_torch(raw_tensor,
                                    dst_tensor,
                                    self.gaussian_kernel_window,
                                    self.downsampling,
                                    self.get_ssim_map,
                                    self.get_cs_map,
                                    self.get_weight,
                                    self.only_test_y_channel,
                                    **self.kwargs)

        return ssim_metrics
    


class Summary(Enum):
    """Enumeration for specifying the type of summary statistics"""
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value
    
    Args:
        name (str): Name of the meter
        fmt (str): Format string for printing
        summary_type (Summary): Type of summary statistic to report
    """
    def __init__(self, name: str, fmt: str = ":f", summary_type: Summary = Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self) -> None:
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics
        
        Args:
            val: Value to update with
            n: Number of items this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self) -> None:
        """All-reduce operation for distributed training"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """String representation"""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        """Summary string"""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.4f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.4f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.4f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Displays training progress
    
    Args:
        num_batches (int): Number of batches in an epoch
        meters (List[AverageMeter]): List of AverageMeter objects to display
        prefix (str): String to prefix the output with
    """
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """Display current batch statistics
        
        Args:
            batch (int): Current batch index
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self) -> None:
        """Display summary statistics"""
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """Get batch format string
        
        Args:
            num_batches (int): Number of batches in an epoch
            
        Returns:
            str: Formatted string for batch display
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    


def build_iqa_model(
        scale: int,
        only_test_y_channel: bool,
        device: torch.device
) -> Tuple[Any, Any]:
    """Build image quality assessment models (PSNR and SSIM)
    
    Args:
        scale: Super-resolution scale factor
        only_test_y_channel: Whether to test on Y channel only
        device: Device to place models on
    
    Returns:
        Tuple of (psnr_model, ssim_model)
    """
    
    
    # Create models with appropriate parameters
    psnr_model = PSNR(
        crop_border=scale,  # Crop border according to scale
        only_test_y_channel=only_test_y_channel,
        data_range=1.0  # Normalized image range
    )
    
    ssim_model = SSIM(
        crop_border=scale,
        only_test_y_channel=only_test_y_channel,
        data_range=1.0  # Normalized image range
    )

    # Move models to device
    psnr_model = psnr_model.to(device)
    ssim_model = ssim_model.to(device)

    return psnr_model, ssim_model
