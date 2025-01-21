"""This file contains perceptual loss module using LPIPS and ConvNeXt-S.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import torch
import torch.nn.functional as F

from torchvision import models
from .lpips import LPIPS

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

 
class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "convnext_s"):
        """Initializes the PerceptualLoss class.

        Args:
            model_name: A string, the name of the perceptual loss model to use.

        Raise:
            ValueError: If the model_name does not contain "lpips" or "convnext_s".
        """
        super().__init__()
        if ("lpips" not in model_name) and (
            "convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the 
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS().eval()

        if "convnext_s" in model_name:
            self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split('-')[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(loss_config[0]), float(loss_config[1])
            print(f"self.loss_weight_lpips, self.loss_weight_convnext: {self.loss_weight_lpips}, {self.loss_weight_convnext}")

        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.eval()
        loss = 0.
        num_losses = 0.
        lpips_loss = 0.
        convnext_loss = 0.
        # Computes LPIPS loss, if available.
        if self.lpips is not None:
            lpips_loss = self.lpips(input, target)
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            # Computes ConvNeXt-s loss, if available.
            input = torch.nn.functional.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
            target = torch.nn.functional.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
            pred_input = self.convnext((input - self.imagenet_mean) / self.imagenet_std)
            pred_target = self.convnext((target - self.imagenet_mean) / self.imagenet_std)
            convnext_loss = torch.nn.functional.mse_loss(
                pred_input,
                pred_target,
                reduction="mean")
                
            if self.loss_weight_convnext is None:
                num_losses += 1
                loss += convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss += self.loss_weight_convnext * convnext_loss
        
        # weighted avg.
        loss = loss / num_losses
        return loss