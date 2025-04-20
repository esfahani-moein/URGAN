import os
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

from .feature_extractor import _FeatureExtractor

class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            net_cfg_name: str = "vgg19",
            batch_norm: bool = False,
            num_classes: int = 1000,
            model_weights_path: str = "",
            feature_nodes: list = None,
            feature_normalize_mean: list = None,
            feature_normalize_std: list = None,
    ) -> None:
        super(ContentLoss, self).__init__()
        # Define the feature extraction model
        model = _FeatureExtractor(net_cfg_name, batch_norm, num_classes)
        # Load the pre-trained model
        if model_weights_path == "":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif model_weights_path is not None and os.path.exists(model_weights_path):
            # Check if file is in safetensors format
            if model_weights_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    checkpoint = load_file(model_weights_path)
                    print(f"Loaded {model_weights_path} using safetensors")
                except ImportError:
                    print("safetensors module not found, falling back to torch.load")
                    checkpoint = torch.load(
                        model_weights_path, 
                        map_location=lambda storage, loc: storage,
                        weights_only=False  # Required for PyTorch 2.6 compatibility
                    )
            else:
                # For PyTorch 2.6 compatibility, set weights_only=False explicitly
                checkpoint = torch.load(
                    model_weights_path, 
                    map_location=lambda storage, loc: storage,
                    weights_only=False
                )
            
            # Load state dict from checkpoint
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
                
            print(f"Successfully loaded weights from {model_weights_path}")
        else:
            raise FileNotFoundError(f"Model weight file not found: {model_weights_path}")
            
        # Extract the output of the feature extraction layer
        self.feature_extractor = create_feature_extractor(model, feature_nodes)
        # Select the specified layers as the feature extraction layer
        self.feature_extractor_nodes = feature_nodes
        # input normalization
        self.normalize = transforms.Normalize(feature_normalize_mean, feature_normalize_std)
        # Freeze model parameters without derivatives
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        device = sr_tensor.device

        losses = []
        # input normalization
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        # Get the output of the feature extraction layer
        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        # Compute feature loss
        for i in range(len(self.feature_extractor_nodes)):
            losses.append(F_torch.l1_loss(sr_feature[self.feature_extractor_nodes[i]],
                                          gt_feature[self.feature_extractor_nodes[i]]))

        losses = torch.Tensor([losses]).to(device)

        return losses