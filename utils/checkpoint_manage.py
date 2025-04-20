import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Import safetensors if available
try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("SafeTensors not found. Install with: pip install safetensors")


class CheckpointManager:
    """Simplified checkpoint manager with SafeTensors support for H100 optimization."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        use_safetensors: bool = True
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            use_safetensors: Whether to use safetensors format (faster loading)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.use_safetensors = use_safetensors and SAFETENSORS_AVAILABLE
        
        # Keep track of saved checkpoints for rotation
        self.checkpoint_paths = []
        
        # Find existing checkpoints
        self._scan_existing_checkpoints()
    
    def _scan_existing_checkpoints(self):
        """Scan checkpoint directory for existing checkpoints."""
        extensions = [".safetensors", ".pt", ".pth", ".tar"]
        for ext in extensions:
            self.checkpoint_paths.extend(
                [str(p) for p in self.checkpoint_dir.glob(f"*{ext}") 
                 if not p.name.endswith(f"_best{ext}")]  # Skip best models
            )
    
    def save(
        self,
        state_dict: Dict[str, Any],
        filename: str,
        is_best: bool = False
    ):
        """Save checkpoint with configured format.
        
        Args:
            state_dict: Dictionary of state to save
            filename: Filename to save as
            is_best: Whether to also save as best model
        """
        # Ensure filename has correct extension
        if self.use_safetensors and not filename.endswith(".safetensors"):
            filename = f"{filename.rsplit('.', 1)[0] if '.' in filename else filename}.safetensors"
        elif not self.use_safetensors and not any(filename.endswith(ext) for ext in [".pt", ".pth", ".tar"]):
            filename = f"{filename}.pt"
            
        # Create full path
        filepath = self.checkpoint_dir / filename
        
        # Save the checkpoint
        if self.use_safetensors:
            self._save_safetensors(state_dict, str(filepath))
        else:
            torch.save(state_dict, filepath)
            
        # Add to tracked checkpoints
        self.checkpoint_paths.append(str(filepath))
        
        # Save best model if requested
        if is_best:
            best_path = self.checkpoint_dir / f"{filename.rsplit('.', 1)[0]}_best.{'safetensors' if self.use_safetensors else 'pt'}"
            if self.use_safetensors:
                self._save_safetensors(state_dict, str(best_path))
            else:
                torch.save(state_dict, best_path)
                
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def _save_safetensors(self, state_dict: Dict[str, Any], filepath: str):
        """Save checkpoint using safetensors format.
        
        Args:
            state_dict: Dictionary of state to save
            filepath: Path to save to
        """
        # SafeTensors can only save tensors, so we need to separate tensor and non-tensor data
        tensor_state = {}
        metadata = {}
        
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                tensor_state[k] = v
            else:
                # Store non-tensor data as metadata
                try:
                    metadata[k] = str(v)
                except:
                    print(f"Warning: Could not serialize {k} as metadata")
        
        # Save the checkpoint with metadata
        safetensors.torch.save_file(tensor_state, filepath, metadata=metadata)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if we have more than max_checkpoints."""
        if len(self.checkpoint_paths) <= self.max_checkpoints:
            return
            
        # Sort checkpoints by creation time
        checkpoint_paths_with_time = []
        for path in self.checkpoint_paths:
            if os.path.exists(path):
                checkpoint_paths_with_time.append((path, os.path.getctime(path)))
        
        checkpoint_paths_with_time.sort(key=lambda x: x[1])  # Sort by creation time
        
        # Remove the oldest checkpoints
        for path, _ in checkpoint_paths_with_time[:-self.max_checkpoints]:
            try:
                os.remove(path)
                self.checkpoint_paths.remove(path)
                print(f"Removed old checkpoint: {os.path.basename(path)}")
            except Exception as e:
                print(f"Failed to remove checkpoint {path}: {e}")
    
    def load(
        self,
        path: str,
        map_location: Union[str, torch.device] = "cpu"
    ) -> Dict[str, Any]:
        """Load checkpoint from file.
        
        Args:
            path: Path to checkpoint
            map_location: Device to load tensors to
            
        Returns:
            Loaded checkpoint state dictionary
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return {}
            
        print(f"Loading checkpoint: {path}")
        
        # Load based on file extension
        if path.endswith(".safetensors"):
            return self._load_safetensors(path, map_location)
        else:
            return torch.load(path, map_location=map_location)
    
    def _load_safetensors(self, path: str, map_location: Union[str, torch.device]) -> Dict[str, Any]:
        """Load checkpoint using safetensors format.
    
        Args:
            path: Path to checkpoint
            map_location: Device to load tensors to
            
        Returns:
            Loaded state dictionary with both tensors and metadata
        """
    # Load tensors without specifying device
        try:
            tensors = safetensors.torch.load_file(path)
            
            # Move tensors to the correct device after loading
            if map_location is not None:
                for k in tensors:
                    if isinstance(tensors[k], torch.Tensor):
                        tensors[k] = tensors[k].to(map_location)
                        
            # Try to get metadata from safetensors
            # Instead of trying to load with "meta" device, use the metadata() function
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                metadata = f.metadata()
                
            # Add metadata to tensors dictionary if available
            if metadata:
                for k, v in metadata.items():
                    tensors[f"_metadata_{k}"] = v
                    
            return tensors
            
        except Exception as e:
            print(f"Error loading safetensors file: {e}")
            # Fall back to loading with PyTorch
            return torch.load(path, map_location=map_location)
        
    def find_latest(self) -> str:
        """Find the latest checkpoint in the directory.
        
        Returns:
            Path to latest checkpoint or empty string if none found
        """
        checkpoint_paths_with_time = []
        for path in self.checkpoint_paths:
            if os.path.exists(path):
                checkpoint_paths_with_time.append((path, os.path.getctime(path)))
                
        if not checkpoint_paths_with_time:
            return ""
            
        # Sort by creation time (newest first)
        checkpoint_paths_with_time.sort(key=lambda x: x[1], reverse=True)
        return checkpoint_paths_with_time[0][0]
    
    def find_best(self) -> str:
        """Find the best checkpoint in the directory.
        
        Returns:
            Path to best checkpoint or empty string if none found
        """
        extensions = [".safetensors", ".pt", ".pth", ".tar"]
        for ext in extensions:
            best_checkpoints = list(self.checkpoint_dir.glob(f"*_best{ext}"))
            if best_checkpoints:
                return str(best_checkpoints[0])
        return ""


# Helper functions for common operations
def save_checkpoint(
    checkpoint_manager: CheckpointManager,
    model: nn.Module,
    ema_model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int = 0,
    psnr: float = 0.0,
    ssim: float = 0.0,
    is_best: bool = False,
    name: str = "model"
):
    """Save checkpoint with all necessary components.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        model: Model to save
        ema_model: EMA model to save (optional)
        optimizer: Optimizer to save (optional)
        scheduler: LR scheduler to save (optional)
        epoch: Current epoch number
        psnr: Current PSNR value
        ssim: Current SSIM value
        is_best: Whether this is the best model so far
        name: Base name for the checkpoint
    """
    # Create state dict
    checkpoint = {
        "epoch": epoch,
        "psnr": psnr,
        "ssim": ssim,
        "model_state_dict": model.state_dict()
    }
    
    # Add EMA model if provided
    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()
    
    # Add optimizer and scheduler if provided
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Format filename with metrics and timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_epoch{epoch}_{timestamp}"
    
    # Save checkpoint
    checkpoint_manager.save(checkpoint, filename, is_best)


def load_checkpoint(
    checkpoint_manager: CheckpointManager,
    path: str,
    model: nn.Module,
    ema_model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Union[str, torch.device] = "cuda"
) -> Dict[str, Any]:
    """Load checkpoint and apply to model and optimizer.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        path: Path to checkpoint
        model: Model to load weights into
        ema_model: EMA model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to
        
    Returns:
        Dictionary with checkpoint metadata
    """
    # Load checkpoint
    checkpoint = checkpoint_manager.load(path, map_location=device)
    if not checkpoint:
        return {}
    
    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load EMA model if available
    if ema_model is not None and "ema_model_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "psnr": checkpoint.get("psnr", 0.0),
        "ssim": checkpoint.get("ssim", 0.0),
    }


def load_latest_checkpoints(config,device,                                   
                                   g_model,d_model,g_optimizer,g_scheduler,d_optimizer,d_scheduler,ema_g_model):

    """Load the latest checkpoint for the generator model."""
    g_checkpoint_dir = os.path.join("checkpoints", f"{config.exp_name}_generator")
    d_checkpoint_dir = os.path.join("checkpoints", f"{config.exp_name}_discriminator")
    
    g_checkpoint_manager = CheckpointManager(
        checkpoint_dir=g_checkpoint_dir,
        max_checkpoints=config.train.checkpoint.max_checkpoints,
        use_safetensors=True
    )
    
    d_checkpoint_manager = CheckpointManager(
        checkpoint_dir=d_checkpoint_dir,
        max_checkpoints=config.train.checkpoint.max_checkpoints,
        use_safetensors=True
    )
    
    try:
        print(f"Attempting to load pretrained generator from: {config.train.checkpoint.pretrained_g_model}")
        
        # First check if it's in our checkpoint manager format
        g_checkpoint_info = load_checkpoint(
            checkpoint_manager=g_checkpoint_manager,
            path=config.train.checkpoint.pretrained_g_model,
            model=g_model,
            device=device
        )
        print(f"Successfully loaded pretrained generator from GAN checkpoint: {config.train.checkpoint.pretrained_g_model}")
    except:
        # If not found, try to load from the RRDBNet checkpoint (from model1 training)
        print(f"GAN checkpoint not found, attempting to load from RRDBNet checkpoint")
        
        # Initialize RRDBNet checkpoint manager
        rrdb_checkpoint_dir = config.train.checkpoint.RRDBNet_trained_path
        rrdb_checkpoint_manager = CheckpointManager(
            checkpoint_dir=rrdb_checkpoint_dir,
            max_checkpoints=config.train.checkpoint.max_checkpoints,
            use_safetensors=True
        )

        try:
            # Try to load best checkpoint first
            best_path = os.path.join(rrdb_checkpoint_dir, "USRGAN_model1_final_epoch200_psnr28.55_ssim0.safetensors")
            g_checkpoint_info = load_checkpoint(
                checkpoint_manager=rrdb_checkpoint_manager,
                path=best_path,
                model=g_model,
                device=device
            )
            print(f"Successfully loaded RRDBNet best checkpoint")
        except (FileNotFoundError, RuntimeError) as e:
            # Try final checkpoint as backup
            try:
                final_path = os.path.join(rrdb_checkpoint_dir, "USRGAN_model1_final_epoch200_psnr28.55_ssim0.safetensors")
                g_checkpoint_info = load_checkpoint(
                    checkpoint_manager=rrdb_checkpoint_manager,
                    path=final_path,
                    model=g_model,
                    device=device
                )
                print(f"Successfully loaded RRDBNet final checkpoint")
            except (FileNotFoundError, RuntimeError) as e:
                print(f"Failed to load RRDBNet checkpoint: {e}")
                raise ValueError("Failed to load required pretrained generator")  


       # Load pretrained discriminator if available
    if config.train.checkpoint.pretrained_d_model:
        try:
            d_checkpoint_info = load_checkpoint(
                checkpoint_manager=d_checkpoint_manager,
                path=config.train.checkpoint.pretrained_d_model,
                model=d_model,
                device=device
            )
            print(f"Loaded pretrained discriminator: {config.train.checkpoint.pretrained_d_model}")
        except Exception as e:
            print(f"Failed to load pretrained discriminator: {e}")
            print("Starting with randomly initialized discriminator")
    else:
        print("No pretrained discriminator specified. Starting with randomly initialized discriminator.")
    

    # Resume training from checkpoint
    if config.train.checkpoint.resumed_g_model:
        g_checkpoint_info = load_checkpoint(
            checkpoint_manager=g_checkpoint_manager,
            path=config.train.checkpoint.resumed_g_model,
            model=g_model,
            ema_model=ema_g_model,
            optimizer=g_optimizer,
            scheduler=g_scheduler,
            device=device
        )
        start_epoch = g_checkpoint_info.get("epoch", 0)
        best_psnr = g_checkpoint_info.get("psnr", 0.0)
        best_ssim = g_checkpoint_info.get("ssim", 0.0)
        print(f"Resumed generator from epoch {start_epoch}, PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f}")
    else:
        print("No generator checkpoint for resuming found. Starting from scratch.")
        
    if config.train.checkpoint.resumed_d_model:
        d_checkpoint_info = load_checkpoint(
            checkpoint_manager=d_checkpoint_manager,
            path=config.train.checkpoint.resumed_d_model,
            model=d_model,
            optimizer=d_optimizer,
            scheduler=d_scheduler,
            device=device
        )
        print(f"Resumed discriminator from checkpoint: {config.train.checkpoint.resumed_d_model}")
    else:
        print("No discriminator checkpoint for resuming found. Starting from scratch.")

    return g_checkpoint_manager, d_checkpoint_manager
           