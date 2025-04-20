
import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# noqa: E402
# pylint: disable=import-error
# type: ignore
try:
    from configs import config_img_proc
except ImportError:
    from ..configs import config_img_proc


extensions= ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

def analyze_image_directory(directory, extensions = extensions, sample_size=100):
    """
    Analyze images in the directory to provide statistics about counts, sizes, and formats
    
    Args:
        directory: Path to the directory containing images
        extensions: List of valid file extensions
        sample_size: Number of images to sample for detailed analysis (default: 100)
    
    Returns:
        Dictionary containing analysis results
    """
    print(f"Analyzing image directory: {directory}")
    
    # Find all image files
    image_files = find_image_files(directory, extensions)
    total_files = len(image_files)
    
    if total_files == 0:
        print("No images found in directory")
        return None
    
    # Count file extensions
    extensions_count = Counter([Path(f).suffix.lower() for f in image_files])
    
    # Sample images for detailed analysis (all if less than sample_size)
    sample_count = min(sample_size, total_files)
    
    # Use random sampling if we have more images than sample_size
    if total_files > sample_size:
        import random
        sample_indices = random.sample(range(total_files), sample_count)
        sample_files = [image_files[i] for i in sample_indices]
    else:
        sample_files = image_files
    
    # Analyze dimensions
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []  # in MB
    
    for img_file in tqdm(sample_files, desc="Sampling images"):
        img_path = Path(directory) / img_file
        
        # Get file size
        file_sizes.append(img_path.stat().st_size / (1024 * 1024))  # Convert to MB
        
        # Get dimensions
        try:
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                height, width = image.shape[:2]
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    # Calculate statistics
    avg_width = sum(widths) / len(widths) if widths else 0
    avg_height = sum(heights) / len(heights) if heights else 0
    avg_filesize = sum(file_sizes) / len(file_sizes) if file_sizes else 0
    
    # Total size estimate
    estimated_total_size_mb = avg_filesize * total_files
    estimated_total_size_gb = estimated_total_size_mb / 1024
    
    # Results dictionary
    results = {
        "total_files": total_files,
        "formats": dict(extensions_count),
        "dimensions": {
            "width": {
                "min": min(widths) if widths else 0,
                "max": max(widths) if widths else 0,
                "avg": avg_width,
            },
            "height": {
                "min": min(heights) if heights else 0,
                "max": max(heights) if heights else 0,
                "avg": avg_height,
            },
            "aspect_ratio": {
                "min": min(aspect_ratios) if aspect_ratios else 0,
                "max": max(aspect_ratios) if aspect_ratios else 0,
                "avg": sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 0,
            },
        },
        "file_size": {
            "min_mb": min(file_sizes) if file_sizes else 0,
            "max_mb": max(file_sizes) if file_sizes else 0,
            "avg_mb": avg_filesize,
            "estimated_total_gb": estimated_total_size_gb,
        },
    }
    
    # Print summary
    print(f"\nImage Directory Analysis Summary:")
    print(f"Total files: {total_files}")
    print(f"File formats: {dict(extensions_count)}")
    print(f"Average dimensions: {avg_width:.1f}x{avg_height:.1f} pixels")
    print(f"Dimension range: {min(widths) if widths else 0}x{min(heights) if heights else 0} to {max(widths) if widths else 0}x{max(heights) if heights else 0}")
    print(f"Average file size: {avg_filesize:.2f} MB")
    print(f"Estimated total size: {estimated_total_size_gb:.2f} GB")
    
    # Memory requirement estimation for GPU processing
    avg_pixels = avg_width * avg_height
    single_image_memory = avg_pixels * 3 * 4 / 1024 / 1024  # RGB float32 in MB
    batch_memory = single_image_memory * config_img_proc.batch_size
    
    print(f"\nGPU Memory Estimates:")
    print(f"Average memory per image (float32): {single_image_memory:.2f} MB")
    print(f"Batch memory requirement: {batch_memory:.2f} MB")
    
    return results



def find_image_files(directory, extensions):
    """Find all image files in a directory (recursively)"""
    # Rest of the function stays the same
    image_files = []
    
    for root, _, files in os.walk(directory):
        rel_path = os.path.relpath(root, directory)
        rel_path = "" if rel_path == "." else rel_path
        
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                if rel_path:
                    image_files.append(os.path.join(rel_path, file))
                else:
                    image_files.append(file)
                    
    return image_files
