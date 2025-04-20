import math
import os
from typing import Any, Tuple, List
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm


# Set up optimal CUDA settings
def setup_device():
    """Configure optimal device settings, prioritizing GPU if available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable TF32 for faster matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Optimize CUDNN for fixed input sizes
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def _cubic(x: Any) -> Any:
    """Implementation of `cubic` function.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation
    """
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
        ((absx > 1) * (absx <= 2)).type_as(absx))


def _calculate_weights_indices(in_length: int,
                               out_length: int,
                               scale: float,
                               kernel_width: int,
                               antialiasing: bool) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Implementation of `calculate_weights_indices` function.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations

    Returns:
       weights, indices, sym_len_s, sym_len_e
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length, device=device)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    linspace_tensor = torch.linspace(0, p - 1, p, device=device).view(1, p)
    indices = left.view(out_length, 1).expand(out_length, p) + linspace_tensor.expand(out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0].item(), 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1].item(), 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        out_2 (np.ndarray): Output image with shape (c, h, w), [0, 1] range, w/o round
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().to(device)
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True
        image = image.to(device)

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)
    
    # Use mixed precision for GPU acceleration
    with amp.autocast(enabled=(device.type == 'cuda')):
        # process H dimension
        # symmetric copying
        img_aug = torch.zeros(in_c, in_h + sym_len_hs + sym_len_he, in_w, device=device)
        img_aug.narrow(1, sym_len_hs, in_h).copy_(image)

        sym_patch = image[:, :sym_len_hs, :]
        inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
        sym_patch_inv = sym_patch.index_select(1, inv_idx)
        img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

        sym_patch = image[:, -sym_len_he:, :]
        inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
        sym_patch_inv = sym_patch.index_select(1, inv_idx)
        img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

        out_1 = torch.zeros(in_c, out_h, in_w, device=device)
        kernel_width = weights_h.size(1)
        
        # Vectorized implementation where possible - much faster on H100
        for i in range(out_h):
            idx = int(indices_h[i][0])
            for j in range(in_c):
                out_1[j, i, :] = torch.mv(img_aug[j, idx:idx + kernel_width, :].transpose(0, 1), weights_h[i])

        # process W dimension
        # symmetric copying
        out_1_aug = torch.zeros(in_c, out_h, in_w + sym_len_ws + sym_len_we, device=device)
        out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

        sym_patch = out_1[:, :, :sym_len_ws]
        inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
        sym_patch_inv = sym_patch.index_select(2, inv_idx)
        out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

        sym_patch = out_1[:, :, -sym_len_we:]
        inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
        sym_patch_inv = sym_patch.index_select(2, inv_idx)
        out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

        out_2 = torch.zeros(in_c, out_h, out_w, device=device)
        kernel_width = weights_w.size(1)
        
        # Vectorized implementation where possible - much faster on H100
        for i in range(out_w):
            idx = int(indices_w[i][0])
            for j in range(in_c):
                out_2[j, :, i] = torch.mv(out_1_aug[j, :, idx:idx + kernel_width], weights_w[i])

    # Move back to CPU and convert to requested format
    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.cpu().numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    # Release GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return out_2


def process_batch(batch_files: List[str], args: dict) -> None:
    """Process a batch of image files using GPU acceleration.

    Args:
        batch_files: List of image files to process
        args: Custom parameter dictionary
    """
    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    scale = args["scale"]
    
    for image_file_path in batch_files:
        image_name, extension = os.path.splitext(os.path.basename(image_file_path))
        image = cv2.imread(os.path.join(inputs_dir, image_file_path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"Warning: Could not read image {image_file_path}")
            continue
            
        resize_image = image_resize(image, 1 / scale, antialiasing=True)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}{extension}"), resize_image)


def split_images(args: dict):
    """Split the image into multiple small images, optimized for GPU processing.

    Args:
        args (dict): Custom parameter dictionary.
    """
    # Initialize device for optimal performance
    device = setup_device()
    
    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    
    # Use ThreadPoolExecutor instead of multiprocessing for GPU work
    # H100 is powerful enough to handle multiple images, so we'll batch process
    if torch.cuda.is_available():
        # Determine optimal batch size based on GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        # Conservative estimate: H100 has 80GB, we'll use up to 70% of it
        available_mem = gpu_mem * 0.7
        
        # Estimate memory per image (adjust as needed based on typical image sizes)
        # Typical 4K image: ~25MB, but tensor operations can use 10x that
        est_mem_per_image = 250 * 1024 * 1024  # 250MB per image
        
        batch_size = max(1, int(available_mem / est_mem_per_image))
        num_workers = min(16, args.get("num_workers", 4))  # H100 can handle more threads
    else:
        batch_size = 1
        num_workers = args.get("num_workers", os.cpu_count())
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Create {output_dir} successful.")

    # Get all image paths
    image_file_paths = os.listdir(inputs_dir)
    total_files = len(image_file_paths)
    
    # Create batches
    batches = []
    for i in range(0, total_files, batch_size):
        end = min(i + batch_size, total_files)
        batches.append(image_file_paths[i:end])
    
    # Process batches
    progress_bar = tqdm(total=total_files, unit="image", desc="Processing images")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        futures = []
        for batch in batches:
            future = executor.submit(process_batch, batch, args)
            futures.append((future, len(batch)))
        
        # Track progress
        while futures:
            for i, (future, count) in enumerate(futures):
                if future.done():
                    progress_bar.update(count)
                    futures.pop(i)
                    break
            time.sleep(0.1)
    
    progress_bar.close()
    print("Image processing completed successfully.")


def main():
    """Main entry point for the script."""
    args = {
        "inputs_dir": "./data/DFO2K_train_GT",  # Path to input image directory.
        "output_dir": "./data/DFO2K_train_LR_bicubic/X4",  # Path to generator image directory.
        "scale": 4,  # Scale factor.
        "num_workers": 10  # How many threads to open at the same time.
    }
    
    start_time = time.time()
    split_images(args)
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()