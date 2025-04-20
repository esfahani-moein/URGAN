import torch
from torch.utils.data import DataLoader

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_fetch.class_dataset import SRDataset
from data_fetch.class_dataloader import load_dataset

from configs import config_dataset



def main():
    


    # Set up device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine if we should use distributed training
    distributed = torch.cuda.device_count() > 1
    print(f"Distributed GPU count: {torch.cuda.device_count()}")
    print(f"Distributed training: {distributed}")

    print(config_dataset)
    print("Loading dataset...")
    # Create dataloaders with train/test split
    train_dataloader, test_dataloader = load_dataset(config_dataset, device, distributed)
    
    # Print dataloader information
    print(f"Training dataloader: {len(train_dataloader)} batches")
    if test_dataloader:
        print(f"Testing dataloader: {len(test_dataloader)} batches")
    else:
        print("No test dataloader created (test_split=0)")
    
    

if __name__ == "__main__":
    main()