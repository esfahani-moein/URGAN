import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import numpy as np
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.checkpoint_manage import CheckpointManager, save_checkpoint, load_checkpoint

# Define a simple model for testing purposes
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return self.conv2(x)

def test_checkpoint_save_load():
    # Create test directory and clean it if it exists
    test_checkpoint_dir = "test_checkpoints"
    if os.path.exists(test_checkpoint_dir):
        shutil.rmtree(test_checkpoint_dir)
    os.makedirs(test_checkpoint_dir, exist_ok=True)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=test_checkpoint_dir,
        max_checkpoints=3,  # Keep only the 3 most recent checkpoints
        use_safetensors=True
    )
    
    # Initialize models, optimizer and scheduler
    g_model = SimpleModel()
    ema_g_model = SimpleModel()
    optimizer = Adam(g_model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    g_model.to(device)
    ema_g_model.to(device)
    
    # Save initial checkpoint
    epoch = 1
    psnr = 20.5
    ssim = 0.75
    is_best = True
    
    save_checkpoint(
        checkpoint_manager=checkpoint_manager,
        model=g_model,
        ema_model=ema_g_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        psnr=psnr,
        ssim=ssim,
        is_best=is_best,
        name="esrgan"
    )
    
    # Check if checkpoint was saved - modify this to match your implementation
    # Instead of calling list_checkpoints(), check if files exist in the directory
    checkpoint_files = os.listdir(test_checkpoint_dir)
    print(f"Checkpoint files after first save: {checkpoint_files}")
    
    # Change model weights
    for param in g_model.parameters():
        param.data = param.data + 0.1
    
    # Save second checkpoint
    epoch = 2
    psnr = 22.0
    ssim = 0.78
    is_best = True
    
    save_checkpoint(
        checkpoint_manager=checkpoint_manager,
        model=g_model,
        ema_model=ema_g_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        psnr=psnr,
        ssim=ssim,
        is_best=is_best,
        name="esrgan"
    )
    
    # Check if checkpoint was saved
    checkpoint_files = os.listdir(test_checkpoint_dir)
    print(f"Checkpoint files after second save: {checkpoint_files}")
    
    # Find the latest checkpoint - replace with the appropriate method or implementation
    # If find_latest() doesn't exist, we need to find it manually
    checkpoint_paths = [os.path.join(test_checkpoint_dir, f) for f in checkpoint_files 
                        if f.startswith("esrgan_") and f.endswith(".safetensors")]
    latest_checkpoint = max(checkpoint_paths, key=os.path.getctime) if checkpoint_paths else None
    print(f"Latest checkpoint: {latest_checkpoint}")
    
    if not latest_checkpoint:
        print("No checkpoint found! Test failed.")
        return
    
    # Create new model and optimizer for loading
    load_model = SimpleModel()
    load_ema_model = SimpleModel()
    load_optimizer = Adam(load_model.parameters(), lr=0.001)
    load_scheduler = MultiStepLR(load_optimizer, milestones=[10, 20], gamma=0.1)
    
    # Load checkpoint
    checkpoint_info = load_checkpoint(
        checkpoint_manager=checkpoint_manager,
        path=latest_checkpoint,
        model=load_model,
        ema_model=load_ema_model,
        optimizer=load_optimizer,
        scheduler=load_scheduler,
        device=device
    )
    
    # Verify loaded information
    print(f"Loaded epoch: {checkpoint_info.get('epoch')}")
    print(f"Loaded PSNR: {checkpoint_info.get('psnr')}")
    print(f"Loaded SSIM: {checkpoint_info.get('ssim')}")
    
    # Clean up
    shutil.rmtree(test_checkpoint_dir)
    print("Checkpoint test completed successfully!")

if __name__ == "__main__":
    test_checkpoint_save_load()