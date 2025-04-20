
import os
import torch

def set_distributed_GPU(distributed: bool):
    if distributed:
        # Check if environment variables are set
        env_vars_set = all(var in os.environ for var in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"])
        
        if env_vars_set:
            # Use environment variables (for Slurm)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
        else:
            # For single node multi-GPU, set variables manually
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            torch.distributed.init_process_group(backend="nccl", 
                                               world_size=torch.cuda.device_count(),
                                               rank=0)
            local_rank = 0
            rank = 0
        
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main_process = (rank == 0)

    else:
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    return device, is_main_process