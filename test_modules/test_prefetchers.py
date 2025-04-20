import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from data_fetch import (
    BaseImageDataset, PairedImageDataset, 
    PrefetchGenerator, PrefetchDataLoader,
    CPUPrefetcher, CUDAPrefetcher
)
from utils import image_to_tensor


class SimpleTestDataset(Dataset):
    """Simple dataset for testing prefetchers"""
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate a random image and tensor
        # Sleep a bit to simulate disk I/O
        time.sleep(0.01)
        
        # Random image of size 64x64 with 3 channels
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        tensor = image_to_tensor(img.astype(np.float32) / 255.0, False, False)
        
        return {
            "idx": idx,
            "image": tensor,
            "value": torch.tensor([idx], dtype=torch.float32)
        }


def test_standard_dataloader():
    """Test standard PyTorch DataLoader"""
    print("\n=== Testing Standard DataLoader ===")
    dataset = SimpleTestDataset(size=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        # Verify batch content
        assert "idx" in batch
        assert "image" in batch
        assert "value" in batch
        assert batch["image"].shape == (4, 3, 64, 64)
        
        # Simulate some processing
        time.sleep(0.05)
        
        print(f"Batch {batch_idx}: Shape={batch['image'].shape}, Indices={batch['idx'].tolist()}")
    
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print("Standard DataLoader test passed!")
    

def test_prefetch_generator():
    """Test PrefetchGenerator"""
    print("\n=== Testing PrefetchGenerator ===")
    dataset = SimpleTestDataset(size=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Wrap with PrefetchGenerator
    prefetch_gen = PrefetchGenerator(iter(dataloader), num_data_prefetch_queue=2)
    
    start_time = time.time()
    batch_idx = 0
    try:
        while True:
            batch = next(prefetch_gen)
            
            # Verify batch content
            assert "idx" in batch
            assert "image" in batch
            assert "value" in batch
            assert batch["image"].shape == (4, 3, 64, 64)
            
            # Simulate some processing
            time.sleep(0.05)
            
            print(f"Batch {batch_idx}: Shape={batch['image'].shape}, Indices={batch['idx'].tolist()}")
            batch_idx += 1
            
    except StopIteration:
        pass
        
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print("PrefetchGenerator test passed!")


def test_prefetch_dataloader():
    """Test PrefetchDataLoader"""
    print("\n=== Testing PrefetchDataLoader ===")
    dataset = SimpleTestDataset(size=20)
    dataloader = PrefetchDataLoader(
        num_prefetch=2,
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        # Verify batch content
        assert "idx" in batch
        assert "image" in batch
        assert "value" in batch
        assert batch["image"].shape == (4, 3, 64, 64)
        
        # Simulate some processing
        time.sleep(0.05)
        
        print(f"Batch {batch_idx}: Shape={batch['image'].shape}, Indices={batch['idx'].tolist()}")
    
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print("PrefetchDataLoader test passed!")


def test_cpu_prefetcher():
    """Test CPUPrefetcher"""
    print("\n=== Testing CPUPrefetcher ===")
    dataset = SimpleTestDataset(size=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Create CPU prefetcher
    prefetcher = CPUPrefetcher(dataloader)
    
    start_time = time.time()
    batch_idx = 0
    batch = prefetcher.next()
    
    while batch:
        # Verify batch content
        assert "idx" in batch
        assert "image" in batch
        assert "value" in batch
        assert batch["image"].shape == (4, 3, 64, 64)
        
        # Simulate some processing
        time.sleep(0.05)
        
        print(f"Batch {batch_idx}: Shape={batch['image'].shape}, Indices={batch['idx'].tolist()}")
        batch_idx += 1
        
        # Get next batch
        batch = prefetcher.next()
    
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print("CPUPrefetcher test passed!")
    
    # Test reset
    print("Testing prefetcher reset...")
    prefetcher.reset()
    assert prefetcher.next() is not None
    print("Reset successful!")


def test_cuda_prefetcher():
    """Test CUDAPrefetcher"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDAPrefetcher test")
        return
        
    print("\n=== Testing CUDAPrefetcher ===")
    device = torch.device("cuda")
    dataset = SimpleTestDataset(size=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    
    # Create CUDA prefetcher
    prefetcher = CUDAPrefetcher(dataloader, device)
    
    start_time = time.time()
    batch_idx = 0
    batch = prefetcher.next()
    
    while batch:
        # Verify batch content
        assert "idx" in batch
        assert "image" in batch
        assert "value" in batch
        assert batch["image"].shape == (4, 3, 64, 64)
        
        # Verify data is on GPU
        assert batch["image"].device.type == "cuda"
        assert batch["value"].device.type == "cuda"
        
        # Simulate some processing
        time.sleep(0.05)
        
        print(f"Batch {batch_idx}: Shape={batch['image'].shape}, Device={batch['image'].device}, Indices={batch['idx'].tolist()}")
        batch_idx += 1
        
        # Get next batch
        batch = prefetcher.next()
    
    print(f"Time taken: {time.time() - start_time:.2f}s")
    print("CUDAPrefetcher test passed!")
    
    # Test reset
    print("Testing prefetcher reset...")
    prefetcher.reset()
    assert prefetcher.next() is not None
    print("Reset successful!")


def test_real_dataset():
    """Test with a real dataset if available"""
    # Check if we have actual image data to test with
    gt_dir = Path("./data/DFO2K_train_GT")
    lr_dir = Path("./data/DFO2K_train_LR_bicubic/X4")
    
    if gt_dir.exists() and lr_dir.exists() and len(list(gt_dir.glob("*.png"))) > 0:
        print("\n=== Testing with real dataset ===")
        
        # Create dataset and prefetchers
        dataset = BaseImageDataset(str(gt_dir), str(lr_dir), upscale_factor=4)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        
        # Standard dataloader
        start_time = time.time()
        print("Testing standard dataloader...")
        for batch_idx, batch in enumerate(tqdm(dataloader, total=3)):
            if batch_idx >= 3:  # Just test 3 batches
                break
            # Basic checks
            assert "gt" in batch
            assert "lr" in batch
            assert batch["gt"].shape[1] == 3  # RGB channels
        std_time = time.time() - start_time
        
        # With CUDA prefetcher if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            prefetcher = CUDAPrefetcher(dataloader, device)
            
            start_time = time.time()
            print("Testing CUDA prefetcher...")
            batch = prefetcher.next()
            batch_idx = 0
            
            while batch and batch_idx < 3:  # Test 3 batches
                # Basic checks
                assert "gt" in batch
                assert "lr" in batch
                assert batch["gt"].shape[1] == 3  # RGB channels
                assert batch["gt"].device.type == "cuda"
                
                batch = prefetcher.next()
                batch_idx += 1
                
            cuda_time = time.time() - start_time
            
            print(f"Standard loader: {std_time:.2f}s, CUDA prefetcher: {cuda_time:.2f}s")
            print(f"Speedup: {std_time / cuda_time:.2f}x")
        
        print("Real dataset test passed!")
    else:
        print("No real dataset found, skipping real dataset test")


if __name__ == "__main__":
    # Run tests
    test_standard_dataloader()
    test_prefetch_generator()
    test_prefetch_dataloader()
    test_cpu_prefetcher()
    test_cuda_prefetcher()
    test_real_dataset()
    
    print("\nAll tests completed successfully!")