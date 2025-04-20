config_image_processing = {
    "hr_dir": "/data/users4/mesfahani1/project_dataset/datasets/Flickr2K/Flickr2K_HR",  # Directory containing HR images
    "lr_dir": "/data/users4/mesfahani1/project_dataset/datasets/Flickr2K/Flickr2K_LR/X4",  # Directory to save LR images
    "scale": 4,  # Downscaling factor (2, 3, 4, or 8)
    "batch_size": 1,  
    "img_extensions": ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'],
    "jpeg_quality": 100,  # JPEG quality setting (0-100)
    "prefetch_batches": 2  # Number of batches to prefetch
}
