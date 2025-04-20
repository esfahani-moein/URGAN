config_model1_net = {
        # General settings
        "seed": 158,
        "device_id": 0,
        "scale": 4,
        "exp_name": "USRGAN_model1",
        
        # Dataset configuration
        "hr_dir": "/data/users4/mesfahani1/project_dataset/datasets/Flickr2K/Flickr2K_HR",
        "lr_dir": "/data/users4/mesfahani1/project_dataset/datasets/Flickr2K/Flickr2K_LR/X4",
        "test_split": 0.1,  #  validation
        
        # Training hyperparameters
        "batch_size": 20,
        "test_batch_size": 8,
        "num_workers": 10,
        "pin_memory": True,
        "persistent_workers": True,
        "shuffle": True,
        "crop_size": 128,
        
        # Model configuration
        "model": {
            "g_name": "RRDBNet",
            "in_channels": 3,
            "out_channels": 3,
            "channels": 64,
            "growth_channels": 32,
            "num_rrdb": 23,
            "compile": False,  # Use torch.compile for H100 speedup
            
            # EMA model settings
            "ema": {
                "enable": True,
                "decay": 0.999,
            }
        },
        
        # Loss function
        "pixel_loss": {
            "name": "L1Loss",
            "weight": [1.0]
        },
        
        # Optimizer settings
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0
        },
        
        # Learning rate scheduler
        "lr_scheduler": {
            "name": "StepLR",
            "step_size": 200000,
            "gamma": 0.5
        },
        
        # Training settings
        "epochs": 200,
        "gradient_accumulation_steps": 2,  # Effective batch size = batch_size * accumulation_steps
        "gradient_clip_norm": 1.0,
        "print_freq": 100,
        
        # Test settings
        "only_test_y_channel": True,
        
        # Checkpoint settings
        "checkpoint": {
            "pretrained_g_model": None,
            "resumed_g_model": None,
            "max_checkpoints": 10,
            "save_freq": 20,  # Save checkpoint every N epochs
        }
    }