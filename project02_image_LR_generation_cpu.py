import time
import multiprocessing

from configs import config_img_proc
from img_preprocessing import process_images_in_parallel

def main():
    
    # Configure process start method for better cross-platform compatibility
    try:
        # Use 'spawn' for better compatibility across platforms
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # If already set, continue
        pass
    
    args = {
        "inputs_dir": config_img_proc.hr_dir,  # Path to input image directory
        "output_dir": config_img_proc.lr_dir,  # Path to output image directory
        "scale": config_img_proc.scale,  # Scale factor (e.g., 4 means 1/4 of original size)
        "num_workers": 64  # Number of CPU cores to use
    }
    
    start_time = time.time()
    process_images_in_parallel(args)
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()