"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet-B0.

IMPORTANT MEDICAL CONTEXT:
This module provides explainability and visual verification of model focus.
Highlighted regions represent areas receiving stronger model attention for the
predicted DR class; they do NOT prove specific lesion presence or substitute for
clinical diagnosis by an ophthalmologist.

Features:
  - Hooks into EfficientNet-B0 final feature extraction layer (features[8]).
  - Defaults to the model's top predicted class, or accepts a target class.
  - Generates raw normalized heatmaps and overlaid visualizations.
  - Side-by-side visualization generator (Original Fundus | Grad-CAM Overlay).
  - Robust exception handling for invalid files, corrupted images, or missing checkpoints.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


# ── Constants & Preprocessing ──────────────────────────────────────────
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Must match inference transforms in predict.py / evaluate.py
infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Model Integrity & Checkpoint Protection ──────────────────────────
PROTECTED_V1_HASH = "3FA407E505F8653F00BD2CFB997224247E76FAC93C18DDEA7F35A973B0B85D05"


def verify_checkpoint_integrity(checkpoint_path: Union[str, Path]) -> None:
    """
    Verifies that the V1 model checkpoint matches the protected SHA-256 hash.
    Halts execution with RuntimeError if verification fails.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

    if ckpt_path.name == "best_model_combined_v1.pth":
        import hashlib
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual_hash = h.hexdigest().upper()
        if actual_hash != PROTECTED_V1_HASH:
            raise RuntimeError(
                f"SAFETY INTEGRITY ERROR: Model checkpoint '{ckpt_path.name}' hash mismatch!\n"
                f"  Expected: {PROTECTED_V1_HASH}\n"
                f"  Actual:   {actual_hash}\n"
                "Inference blocked: checkpoint appears corrupted or modified."
            )


def load_model(checkpoint_path: Union[str, Path], device: torch.device) -> nn.Module:
    """
    Constructs EfficientNet-B0 with 5-class head and loads checkpoint weights.
    Strictly offline: constructs architecture with weights=None (zero network download).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Model integrity verification before loading weights
    verify_checkpoint_integrity(ckpt_path)

    # Build base model architecture strictly offline (no pretrained weights requested)
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


class GradCAM:
    """
    Computes Grad-CAM visualizations for EfficientNet-B0.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        target_layer: Optional[nn.Module] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(checkpoint_path, self.device)
        
        # Target layer is features[8] (the final 1280-dim Conv2dNormActivation layer)
        self.target_layer = target_layer if target_layer is not None else self.model.features[8]

        self.activations = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where element 0 is the gradient wrt output
            self.gradients.append(grad_out[0])

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Removes PyTorch module hooks when no longer needed."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        image_input: Union[str, Path, Image.Image],
        target_class: Optional[int] = None,
        colormap: str = "jet",
        alpha: float = 0.4
    ) -> Dict[str, Any]:
        """
        Generates Grad-CAM heatmap and visualization overlay.

        Parameters:
            image_input: Path to image or PIL Image object
            target_class: Optional integer (0-4). If None, uses model's top predicted class.
            colormap: Matplotlib colormap name for heatmap rendering (default: 'jet')
            alpha: Heatmap blend factor on original image (0.0 to 1.0)

        Returns:
            Dict containing:
              - predicted_class: int
              - class_name: str
              - confidence: float (0-100)
              - probabilities: list of float
              - target_class: int
              - raw_heatmap: 2D float numpy array (H, W) in range [0, 1]
              - overlay_image: PIL Image of overlay on original input image
              - side_by_side_image: PIL Image with [Original | Overlay]
        """
        self.activations.clear()
        self.gradients.clear()

        # 1. Load and format image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")
            orig_pil = Image.open(img_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
        else:
            raise TypeError("image_input must be a file path or PIL.Image")

        orig_w, orig_h = orig_pil.size

        # 2. Preprocessing
        input_tensor = infer_transforms(orig_pil).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        # 3. Forward pass
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        
        pred_class = int(torch.argmax(probs).item())
        pred_confidence = float(probs[pred_class].item()) * 100.0
        all_probs = [float(p.item()) for p in probs]

        if target_class is None:
            target_class = pred_class
        elif not (0 <= target_class < NUM_CLASSES):
            raise ValueError(f"target_class must be between 0 and {NUM_CLASSES - 1}, got {target_class}")

        # 4. Backward pass for target class score
        score = logits[0, target_class]
        score.backward()

        # 5. Compute Grad-CAM weights & activation map
        # activations: (1, 1280, 7, 7), gradients: (1, 1280, 7, 7)
        acts = self.activations[-1].detach()
        grads = self.gradients[-1].detach()

        # Global average pooling of gradients over spatial dimensions (H=7, W=7)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, 1280, 1, 1)

        # Weighted combination of feature activation maps
        cam = torch.sum(weights * acts, dim=1).squeeze(0)  # (7, 7)

        # Apply ReLU to retain only features with a positive influence
        cam = F.relu(cam).cpu().numpy()

        # Normalize heatmap safely to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        # 6. Resize heatmap to original image dimensions
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_pil.resize((orig_w, orig_h), resample=Image.Resampling.BICUBIC)
        heatmap_norm = np.array(heatmap_resized, dtype=np.float32) / 255.0

        # 7. Apply colormap (compatible with both modern and older matplotlib)
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps[colormap]
        except Exception:
            cmap = cm.get_cmap(colormap)
        colored_cam = cmap(heatmap_norm)[:, :, :3]  # drop alpha, RGB in [0, 1]
        colored_cam = (colored_cam * 255).astype(np.uint8)

        # 8. Overlay onto original fundus image
        orig_arr = np.array(orig_pil, dtype=np.float32)
        blended = (1.0 - alpha) * orig_arr + alpha * colored_cam
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(blended)

        # 9. Create Side-by-Side visualization (Original | Overlay)
        side_by_side = Image.new("RGB", (orig_w * 2, orig_h))
        side_by_side.paste(orig_pil, (0, 0))
        side_by_side.paste(overlay_pil, (orig_w, 0))

        # Clean up model gradients and intermediate buffers after backward pass
        self.model.zero_grad()
        self.activations.clear()
        self.gradients.clear()

        return {
            "predicted_class": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "confidence": round(pred_confidence, 2),
            "probabilities": [round(p * 100, 2) for p in all_probs],
            "raw_probabilities": all_probs,
            "target_class": target_class,
            "target_class_name": CLASS_NAMES[target_class],
            "raw_heatmap": heatmap_norm,
            "heatmap_size": (orig_w, orig_h),
            "overlay_image": overlay_pil,
            "side_by_side_image": side_by_side,
        }

    def save_visualization(
        self,
        result: Dict[str, Any],
        output_path: Union[str, Path],
        mode: str = "side_by_side"
    ) -> Path:
        """
        Saves overlay or side-by-side image to disk.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img_to_save = result["side_by_side_image"] if mode == "side_by_side" else result["overlay_image"]
        img_to_save.save(out)
        return out
