import os
import time
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch

# Configuration parameters - optimized for GPU processing
CONFIG = {
    "hr_dir": "./data/DIV2K_train_HR",  # Directory containing HR images
    "lr_dir": "./data/DIV2K_train_LR_bicubic/X4",  # Directory to save LR images
    "scale": 4,  # Downscaling factor (2, 3, 4, or 8)
    "batch_size": 32,  # Increased for A100 GPU
    "img_extensions": ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'],
    "jpeg_quality": 95,  # JPEG quality setting (0-100)
    "prefetch_batches": 2  # Number of batches to prefetch
}

def find_image_files(directory, extensions):
    """Find all image files in a directory (recursively)"""
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

class GPUImageProcessor:
    """GPU-accelerated image processing"""
    def __init__(self):
        # Check for GPU and get device
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available. This script requires CUDA.")
        
        self.device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Create downsample layer once and reuse
        self.scale = CONFIG["scale"]
        self.resize_layer = torch.nn.Upsample(
            scale_factor=1/self.scale, 
            mode='bicubic',
            align_corners=False
        ).to(self.device)
        
        # Set up prefetcher variables
        self.prefetch_queue = []
        self.prefetch_batches = CONFIG["prefetch_batches"]
    
    def load_batch(self, image_files, input_dir):
        """Load a batch of images into memory and prepare for GPU"""
        input_images = []
        paths = []
        
        for img_file in image_files:
            img_path = Path(input_dir) / img_file
            output_path = Path(CONFIG["lr_dir"]) / img_file
            
            # Create output subdirectories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if output already exists
            if output_path.exists():
                continue
                
            # Read image
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Drop alpha channel
            
            input_images.append(image)
            paths.append((img_file, output_path))
        
        if not input_images:
            return None
            
        # Convert to tensor
        batch = torch.stack([
            torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            for img in input_images
        ])
        
        return {"images": batch, "paths": paths}
    
    def prefetch_batches(self, batch_indices, all_image_files, input_dir):
        """Prefetch batches in advance to hide I/O latency"""
        self.prefetch_queue = []
        for idx in batch_indices:
            start_idx = idx * CONFIG["batch_size"]
            end_idx = min(start_idx + CONFIG["batch_size"], len(all_image_files))
            batch_files = all_image_files[start_idx:end_idx]
            batch_data = self.load_batch(batch_files, input_dir)
            if batch_data:
                self.prefetch_queue.append(batch_data)
    
    def process_images(self, all_image_files, input_dir):
        """Process all images using GPU batching with prefetching"""
        total_images = len(all_image_files)
        processed_count = 0
        
        with torch.cuda.amp.autocast():  # Use mixed precision for speed
            with torch.no_grad():  # No gradients needed
                with tqdm(total=total_images, desc="Generating LR images", unit="img") as pbar:
                    # Process images in batches
                    batch_count = (total_images + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
                    
                    for batch_idx in range(batch_count):
                        start_idx = batch_idx * CONFIG["batch_size"]
                        end_idx = min(start_idx + CONFIG["batch_size"], total_images)
                        batch_files = all_image_files[start_idx:end_idx]
                        
                        # Load batch data
                        batch_data = self.load_batch(batch_files, input_dir)
                        if not batch_data:
                            continue
                        
                        # Move batch to GPU
                        batch = batch_data["images"].to(self.device)
                        paths = batch_data["paths"]
                        
                        # Process batch (downsample images)
                        lr_batch = self.resize_layer(batch)
                        
                        # Move results back to CPU
                        lr_batch = lr_batch.cpu().numpy()
                        
                        # Save each image
                        for i, (_, output_path) in enumerate(paths):
                            # Convert back to uint8 for OpenCV
                            lr_img = (lr_batch[i].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                            
                            # Save with appropriate quality settings
                            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                                cv2.imwrite(str(output_path), lr_img, [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpeg_quality"]])
                            elif output_path.suffix.lower() == '.png':
                                cv2.imwrite(str(output_path), lr_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                            else:
                                cv2.imwrite(str(output_path), lr_img)
                        
                        # Update progress bar
                        processed = len(paths)
                        pbar.update(processed)
                        processed_count += processed
                        
                        # Free memory
                        torch.cuda.empty_cache()
        
        return processed_count

def generate_lr_dataset():
    """Generate a low-resolution dataset from high-resolution images using GPU only"""
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG["lr_dir"], exist_ok=True)
    
    # Find all image files from the HR directory
    image_files = find_image_files(CONFIG["hr_dir"], CONFIG["img_extensions"])
    
    if not image_files:
        print(f"No image files found in {CONFIG['hr_dir']}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Initialize GPU processor
    processor = GPUImageProcessor()
    
    # Process all images
    processed_count = processor.process_images(image_files, CONFIG["hr_dir"])
    
    # Report results
    elapsed = time.time() - start_time
    print(f"Successfully processed {processed_count} images in {elapsed:.2f} seconds")
    if processed_count > 0:
        print(f"Average processing time: {elapsed/processed_count:.4f} seconds per image")

if __name__ == "__main__":
    generate_lr_dataset()