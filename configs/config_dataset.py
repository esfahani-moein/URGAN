config_dataset = {
        # Dataset paths - only one set of directories needed
        "hr_dir": "/data/users4/mesfahani1/project_dataset/datasets/Flickr2K/Flickr2K_HR",
        "lr_dir": "/data/users4/mesfahani1/project_dataset/datasets/Flickr2K/Flickr2K_LR/X4",  # Can be None if generating LR on the fly
        
        # Model parameters
        "scale": 4,
        "crop_size": 128,
        
        # Data split configuration
        "test_split": 0.1,  # 10% of data for testing
        "seed": 42,         # For reproducible splits
        
        # DataLoader parameters
        "batch_size": 32,
        "test_batch_size": 1,
        "num_workers": 10,   # Adjust based on your CPU cores
        "shuffle": True,    # Shuffle training data
    }