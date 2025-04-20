import os
import time
from tqdm import tqdm


from configs import config_img_proc
from img_preprocessing import find_image_files, GPUImageProcessor

def generate_lr_dataset():
    """Generate a low-resolution dataset from high-resolution images"""
    start_time = time.time()
    
    hr_dir = config_img_proc.hr_dir
    lr_dir = config_img_proc.lr_dir
    scale = config_img_proc.scale
    batch_size = config_img_proc.batch_size
    
    print(f"Generating LR images from {hr_dir} at scale x{scale}")
    print(f"Output will be saved to {lr_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(lr_dir, exist_ok=True)
    
    # Find all image files from the HR directory (recursively)
    image_files = find_image_files(hr_dir, config_img_proc.img_extensions)
    
    if not image_files:
        print(f"No image files found in {hr_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # GPU-accelerated batch processing
    if len(image_files) > 1:
        gpu_processor = GPUImageProcessor()
        
        with tqdm(total=len(image_files), desc="Generating LR images (GPU)", unit="img") as pbar:
            # Process images in batches
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i+batch_size]
                gpu_processor.process_batch(batch_files, hr_dir, lr_dir, scale)
                pbar.update(len(batch_files))
    
    elapsed = time.time() - start_time
    print(f"Successfully processed {len(image_files)} images in {elapsed:.2f} seconds")
    print(f"Average processing time: {elapsed/len(image_files):.4f} seconds per image")

if __name__ == "__main__":
    generate_lr_dataset()