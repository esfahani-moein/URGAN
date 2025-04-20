import os
import glob
import random
from PIL import Image
from typing import Dict
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SRDataset(Dataset):
    """Enhanced dataset for ESRGAN training with pre-existing LR and HR pairs"""
    
    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        crop_size: int = 128,
        scale: int = 4,
        augment: bool = True,
        split: str = "train",
        file_extension: str = "png"
    ) -> None:
        """Initialize SR dataset with paired LR and HR images.
        
        Args:
            hr_dir: Directory containing HR images
            lr_dir: Directory containing LR images
            crop_size: Size of HR patches to crop during training
            scale: Super-resolution scale factor
            augment: Whether to apply data augmentation
            split: Dataset split ('train' or 'test')
            file_extension: Image file extension to load
        """
        super(SRDataset, self).__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.scale = scale
        self.augment = augment and split == "train"  # Only augment training data
        self.split = split
        
        # Get image file pairs - assumes matching filenames
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, f"*.{file_extension}")))
        if lr_dir:
            self.lr_files = sorted(glob.glob(os.path.join(lr_dir, f"*.{file_extension}")))
            assert len(self.hr_files) == len(self.lr_files), "HR and LR directories must have same number of images"
        else:
            self.lr_files = None
        
        # For fast image loading
        self._cache = {}
        self._cache_size = 100  # Adjust based on available RAM
    
    def __len__(self) -> int:
        return len(self.hr_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        try:

            # Load images
            hr_path = self.hr_files[idx]
            
            # Use LR from directory if available, otherwise will generate from HR
            if self.lr_files:
                lr_path = self.lr_files[idx]
            else:
                lr_path = None
            
            # Try to load from cache first
            if hr_path in self._cache:
                hr_img = self._cache[hr_path]
            else:
                hr_img = Image.open(hr_path).convert("RGB")
                # Cache with LRU policy
                if len(self._cache) >= self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[hr_path] = hr_img
            
            if lr_path:
                if lr_path in self._cache:
                    lr_img = self._cache[lr_path]
                else:
                    lr_img = Image.open(lr_path).convert("RGB")
                    if len(self._cache) >= self._cache_size:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[lr_path] = lr_img
            else:
                # Generate LR from HR for fallback behavior
                lr_img = hr_img.resize((hr_img.width // self.scale, hr_img.height // self.scale),
                                    Image.BICUBIC)
            
            # Convert to tensors
            hr_tensor = TF.to_tensor(hr_img).clone()
            lr_tensor = TF.to_tensor(lr_img).clone()
            
            # Training mode: crop random patches
            if self.split == "train":
                hr_tensor = TF.resize(hr_tensor, [self.crop_size, self.crop_size], 
                                antialias=True).clone()
                lr_tensor = TF.resize(lr_tensor, [self.crop_size // self.scale, self.crop_size // self.scale],
                                antialias=True).clone()
            
                # Apply augmentations if enabled
                if self.augment:
                    # Random rotation
                    if random.random() > 0.5:
                        angle = random.choice([90, 180, 270])
                        hr_tensor = TF.rotate(hr_tensor, angle).clone()
                        lr_tensor = TF.rotate(lr_tensor, angle).clone()
                    
                    # Random flips
                    if random.random() > 0.5:
                        hr_tensor = TF.vflip(hr_tensor).clone()
                        lr_tensor = TF.vflip(lr_tensor).clone()
                    
                    if random.random() > 0.5:
                        hr_tensor = TF.hflip(hr_tensor).clone()
                        lr_tensor = TF.hflip(lr_tensor).clone()
        
            # Final explicit detach and clone
            return {
                "gt": hr_tensor.detach().clone(),
                "lr": lr_tensor.detach().clone(),
                "hr_path": hr_path,
                "lr_path": lr_path or "generated"
            }
        except Exception as e:
            print(f"Error processing image index {idx}: {e}")
            # Create fresh default tensors with detach and clone
            default_hr = torch.zeros((3, self.crop_size, self.crop_size)).detach().clone()
            default_lr = torch.zeros((3, self.crop_size // self.scale, self.crop_size // self.scale)).detach().clone()
            return {
                "gt": default_hr,
                "lr": default_lr,
                "hr_path": hr_path if 'hr_path' in locals() else f"error_idx_{idx}",
                "lr_path": "error"
            }
        
    def set_split(self, split):
        """Update the split value for this dataset."""
        self.split = split
        self.augment = self.augment and split == "train"