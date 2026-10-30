"""
model.py
========
Model factory and Grad-CAM implementation for ResNet18.

Grad-CAM:
  Uses PyTorch forward/backward hooks on `layer4` (the last conv block of ResNet18).
  No external library; pure PyTorch hooks. Verified against the original paper:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", ICCV 2017.
"""

import io
import base64
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ImageNet stats (must match dataset.py)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet18 with a replaced classification head for `num_classes` outputs.

    Args:
        num_classes: Number of output classes (38 for PlantVillage).
        pretrained: If True, load ImageNet pretrained weights for the backbone.

    Returns:
        torch.nn.Module ready for training or inference.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model_from_file(
    checkpoint_path: str,
    num_classes: int,
    device: str = "cpu",
) -> nn.Module:
    """
    Load a saved model checkpoint (.pth dict with 'model_state_dict' and 'class_to_idx').

    Returns:
        (model, class_to_idx)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    class_to_idx = checkpoint.get("class_to_idx", {})
    return model, class_to_idx


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet18.

    Hooks into `model.layer4` (the last residual block).
    Call `.generate(input_tensor, class_idx)` to get a heatmap.

    Reference:
      Selvaraju et al. ICCV 2017. https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_layer = self.model.layer4  # last conv block

        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Shape (1, 3, H, W), already normalised.
            class_idx: If None, use the predicted class.

        Returns:
            heatmap: float32 numpy array in [0, 1], shape (H, W).
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pooling of gradients → weights
        gradients = self._gradients   # (1, C, H', W')
        activations = self._activations  # (1, C, H', W')

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # keep only positive influence

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.astype(np.float32)


def heatmap_to_base64_png(
    heatmap: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.5,
) -> str:
    """
    Overlay Grad-CAM heatmap on the original image and return as base64-encoded PNG.

    Args:
        heatmap: float32 (H, W) in [0, 1] — the Grad-CAM output.
        original_image: PIL RGB image (will be resized to heatmap size).
        alpha: Overlay transparency.

    Returns:
        Base64-encoded PNG string.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Resize heatmap to original image size
    h, w = original_image.size[1], original_image.size[0]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    # Apply colormap
    try:
        colormap = matplotlib.colormaps["jet"]
    except AttributeError:
        colormap = plt.get_cmap("jet")
    heatmap_rgb = colormap(heatmap_resized)[:, :, :3]  # (H, W, 3) RGB

    orig_arr = np.array(original_image).astype(np.float32) / 255.0
    overlay = (1 - alpha) * orig_arr + alpha * heatmap_rgb
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess a PIL image for model inference. Returns (1, 3, H, W) tensor."""
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),  # e.g. 256 for image_size=224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)  # (1, 3, H, W)
