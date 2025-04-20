
import os
import cv2
import torch
import numpy as np
from pathlib import Path


class GPUImageProcessor:
    """Class to handle GPU-accelerated image processing"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = self.device.type == 'cuda'
        if self.use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available or disabled, using CPU")
    
    def process_batch(self, image_files, input_dir, output_dir, scale):
        """Process a batch of images using GPU"""
        input_images = []
        paths = []
        
        # Load all images in the batch
        for img_file in image_files:
            img_path = Path(input_dir) / img_file
            output_path = Path(output_dir) / img_file
            
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
                
            # Convert to RGB and normalize
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Drop alpha channel
            
            input_images.append(image)
            paths.append((img_file, img_path, output_path))
        
        if not input_images:
            return
            
        # Process images in a batch on GPU
        with torch.no_grad():
            # Convert to tensor and move to GPU
            batch = torch.stack([
                torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                for img in input_images
            ]).to(self.device)
            
            # Define bicubic downsampling operation
            resize_layer = torch.nn.Upsample(
                scale_factor=1/scale, 
                mode='bicubic',
                align_corners=False
            ).to(self.device)
            
            # Apply downsampling
            lr_batch = resize_layer(batch)
            
            # Move back to CPU and convert to numpy for saving
            lr_batch = lr_batch.cpu().numpy()
            
            # Save each image
            for i, (img_file, _, output_path) in enumerate(paths):
                # Convert back to uint8 and BGR for OpenCV
                lr_img = (lr_batch[i].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                
                # Save the image with high quality
                if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                    cv2.imwrite(str(output_path), lr_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif output_path.suffix.lower() == '.png':
                    cv2.imwrite(str(output_path), lr_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    cv2.imwrite(str(output_path), lr_img)

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