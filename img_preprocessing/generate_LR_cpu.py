import os
import time
import cv2
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


def bicubic_kernel(x):
    """Implementation of the bicubic interpolation kernel."""
    x = np.abs(x)
    if x <= 1:
        return 1.5 * x**3 - 2.5 * x**2 + 1
    elif x < 2:
        return -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2
    else:
        return 0


def resize_image(image, scale_factor, antialiasing=True):
    """Resize an image using OpenCV's bicubic interpolation.
    
    Args:
        image: Input image as NumPy array
        scale_factor: Scale factor (0.5 for downsampling by 2x)
        antialiasing: Whether to apply antialiasing
        
    Returns:
        Resized image
    """
    # Calculate target dimensions
    h, w = image.shape[:2]
    target_h, target_w = int(h * scale_factor), int(w * scale_factor)
    
    # OpenCV's resize with INTER_CUBIC provides bicubic interpolation
    # For downsampling with antialiasing, INTER_AREA often gives better results
    if scale_factor < 1.0 and antialiasing:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
        
    resized_img = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    return resized_img


def process_image(image_file_path, args):
    """Process a single image file.
    
    Args:
        image_file_path: Path to the image file
        args: Dictionary of parameters
    """
    try:
        inputs_dir = args["inputs_dir"]
        output_dir = args["output_dir"]
        scale = args["scale"]
        
        image_name, extension = os.path.splitext(os.path.basename(image_file_path))
        full_path = os.path.join(inputs_dir, image_file_path)
        
        image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"Warning: Could not read image {image_file_path}")
            return
            
        # Resize image using bicubic interpolation with antialiasing
        resize_image_result = resize_image(image, 1/scale, antialiasing=True)
        
        output_path = os.path.join(output_dir, f"{image_name}{extension}")
        cv2.imwrite(output_path, resize_image_result)
        
    except Exception as e:
        print(f"Error processing {image_file_path}: {str(e)}")


def process_images_in_parallel(args):
    """Process images using CPU optimization with multiple processes.
    
    Args:
        args: Dictionary of parameters including paths and scale factor
    """
    inputs_dir = args["inputs_dir"]
    output_dir = args["output_dir"]
    num_workers = args.get("num_workers", 32)  # Default to 32 CPU cores
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created {output_dir} successfully.")

    # Get all image paths with image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_file_paths = [f for f in os.listdir(inputs_dir) 
                      if os.path.isfile(os.path.join(inputs_dir, f)) and 
                      f.lower().endswith(valid_extensions)]
    
    total_files = len(image_file_paths)
    
    if total_files == 0:
        print(f"No image files found in {inputs_dir}")
        return
    
    print(f"Processing {total_files} images using {num_workers} CPU workers")
    
    # Process files in parallel
    process_func = partial(process_image, args=args)
    
    # Creating a progress bar
    with tqdm(total=total_files, unit="image") as pbar:
        # Using ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            # Submit all tasks
            for img_path in image_file_paths:
                future = executor.submit(process_func, img_path)
                futures.append(future)
                
            # Monitor completion and update progress bar
            for future in futures:
                future.result()  # Wait for completion (also raises any exceptions)
                pbar.update(1)
    
    print("Image processing completed successfully.")
