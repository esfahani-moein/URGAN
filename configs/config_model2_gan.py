# Configuration file for model.
config_model2_gan = {
    "current_dir": "/data/users4/mesfahani1/project09_super_res",
    "exp_name": "USRGAN_model2_X4",
    "seed": 42,
    "scale": 4,
    "device_id": 0,
    "model": {
        "g": {
            "name": "rrdbnet_x4",
            "in_channels": 3,
            "out_channels": 3,
            "channels": 64,
            "growth_channels": 32,
            "num_rrdb": 23,
            "compiled": False
        },
        "d": {
            "name": "discriminator_for_vgg",
            "in_channels": 3,
            "out_channels": 1,
            "channels": 64,
            "upsample_method": "nearest",
            "compiled": False
        },
        "ema": {
            "enable": True,
            "decay": 0.999,
            "compiled": False
        }
    },
    "train": {
        "dataset": {
            "paired_train_gt_images_dir": "/data/users4/mesfahani1/project_dataset/datasets/Flickr2K/Flickr2K_HR",
            "gt_image_size": 128
        },
        "hyp": {
            "epochs": 100, # ******  6000 - 7000 epochs for 16 img per batch
            "shuffle": True,
            "num_workers": 10,
            "pin_memory": True,
            "persistent_workers": True
        },
        "optim": {
            "name": "Adam",
            "lr": 1e-4,
            "betas": [0.9, 0.99],
            "eps": 1e-8,
            "weight_decay": 0.0
        },
        "lr_scheduler": {
            "name": "MultiStepLR",
            "milestones": [20, 40],
            "gamma": 0.5
        },
        "losses": {
            "pixel_loss": {
                "name": "L1Loss",
                "weight": [1.0]
            },
            "content_loss": {
                "name": "ContentLoss",
                "weight": [1.0],
                "net_cfg_name": "vgg19",
                "batch_norm": False,
                "num_classes": 1000,
                "model_weights_path": "/data/users4/mesfahani1/project09_super_res/checkpoints/pretrained_models/vgg19_imagenet.safetensors",
                "feature_nodes": ["features.2", "features.7", "features.16", "features.25", "features.34"],
                "feature_normalize_mean": [0.485, 0.456, 0.406],
                "feature_normalize_std": [0.229, 0.224, 0.225]
            },
            "adversarial_loss": {
                "name": "vanilla",
                "weight": [0.005]
            }
        },
        "print_freq": 100,
        "checkpoint": {
            "RRDBNet_trained_path": "/data/users4/mesfahani1/project09_super_res/checkpoints/USRGAN_model1",
            "pretrained_g_model": None,
            "pretrained_d_model": None,
            "resumed_g_model": None,
            "resumed_d_model": None,
            "max_checkpoints": 10,
        }
    },
    "test": {
        "dataset": { # ****** 
            "paired_test_gt_images_dir": "",
            "paired_test_lr_images_dir": ""
        },
        "hyp": {
            "imgs_per_batch": 1,
            "shuffle": False,
            "num_workers": 1,
            "pin_memory": True,
            "persistent_workers": True
        },
        "only_test_y_channel": True
    }
}