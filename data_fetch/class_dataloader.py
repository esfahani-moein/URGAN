import os
import random
import torch
from torch.utils.data import DataLoader, random_split

# Import the SRDataset from the dataset module
from .class_dataset import SRDataset
from .custom_collate import custom_collate

def load_dataset(config, device, distributed=False):
    """Create  data loaders.
    
    Args:
        config: Dictionary containing configuration parameters
        device: Device to use for training
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create full dataset
    full_dataset = SRDataset(
        hr_dir=config.hr_dir,
        lr_dir=config.lr_dir,
        crop_size=config.crop_size,
        scale=config.scale,
        augment=True,  # Augmentation will only be applied for training split
        split="train"  # This will be overridden for test set later
    )
    
    # Calculate the split sizes
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * config.test_split)
    train_size = dataset_size - test_size
    
    # Split the dataset
    if test_size > 0:
        # Set seed for reproducibility
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, test_size],
            generator=generator
        )
        
        # Create a new test dataset with explicit test configuration
        test_dataset_explicit = SRDataset(
            hr_dir=config.hr_dir,
            lr_dir=config.lr_dir,
            crop_size=config.crop_size,  # Use same crop size for test - important!
            scale=config.scale,
            augment=False,
            split="test"
        )
        
        # Use the same indices from the random split
        test_dataset_explicit.hr_files = [full_dataset.hr_files[idx] for idx in test_dataset.indices]
        if test_dataset_explicit.lr_files:
            test_dataset_explicit.lr_files = [full_dataset.lr_files[idx] for idx in test_dataset.indices]
        
        # Replace with our explicit test dataset
        test_dataset = test_dataset_explicit
    else:
        train_dataset = full_dataset
        test_dataset = None
    
    print(f"Dataset split: {train_size} training images, {test_size} testing images")
    
    # Setup samplers for distributed training
    train_sampler = None
    test_sampler = None
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=config.shuffle
        )
        if len(test_dataset) > 0:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                shuffle=False
            )
        shuffle = False  # Sampler handles shuffling
    else:
        shuffle = config.shuffle
    
    # Create training dataloader with H100 optimizations
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle and train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,  # Always pin memory for H100
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=3,  # Prefetch 2 batches per worker
        collate_fn=custom_collate,  # Custom collate function
    )
    
    # Create testing dataloader (if we have test data)
    if len(test_dataset) > 0:
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            collate_fn=custom_collate, # Custom collate function
        )
    else:
        test_dataloader = None
    
    return train_dataloader, test_dataloader