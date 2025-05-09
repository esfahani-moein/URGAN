# URGAN - Ultra Resolution Generative Adversarial Network

Efficient Generative Adversarial Networks (GANs) Model for Image Super-resolution

This URGAN model is deep learning project implemented in PyTorch, dedicated to advancing the state-of-the-art in single image super-resolution (SISR). 
Drawing inspiration from the highly successful Real-ESRGAN and the goal is to get better results, URGAN leverages Generative Adversarial Networks (GANs) to reconstruct high-resolution images from low-resolution inputs, focusing on perceptual quality and realistic detail generation.

This repository provides a comprehensive framework for training and evaluating ultra-resolution models, featuring:

Modular PyTorch Implementation: 
compared to real model, It's Clean, organized, and extensible codebase built entirely with PyTorch, facilitating research and development.
this has Two-Stage Training Pipeline: Employs a robust training strategy involving an initial PSNR-oriented pre-training phase followed by GAN-based fine-tuning for enhanced perceptual realism.

Advanced Network Architectures: Incorporates powerful backbone networks like Residual-in-Residual Dense Block (RRDBNet) for the generator, coupled with a U-Net discriminator and VGG-based perceptual loss.


High Configurability: Easily customize datasets, model architectures, loss functions, training hyperparameters, and data augmentation strategies through dedicated configuration files.

GPU-Accelerated Data Preprocessing: Includes optimized scripts for efficient generation of low-resolution training data directly on the GPU, significantly speeding up the data preparation workflow.

Utilities: Offers a set of utility functions for checkpoint management (including SafeTensors support), logging, standard image quality metrics (PSNR, SSIM), and data loading/augmentation.

Reproducibility: Designed with clear structure and configuration management to aid in reproducing experimental results.

URGAN aims to provide researchers and practitioners with a powerful and flexible platform for exploring and developing high-fidelity image super-resolution techniques within the PyTorch ecosystem.
