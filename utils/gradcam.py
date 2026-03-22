"""
============================================
Antigravity - Grad-CAM Visualization Module
============================================
Generates heatmap overlays on chest X-rays to show
which regions the model focused on when making its
pneumonia prediction. Uses torchcam library.
============================================
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit

import torch
import torch.nn.functional as F
from torchvision import transforms


def generate_gradcam(model, input_tensor, original_image, target_class=None):
    """
    Generate a Grad-CAM heatmap for a given input image and model.
    
    This function computes Grad-CAM by:
      1. Finding the last convolutional layer
      2. Registering forward/backward hooks to capture activations & gradients
      3. Running a forward + backward pass
      4. Computing the weighted combination of activation maps
      5. Overlaying the heatmap on the original image
    
    Args:
        model: The PyTorch model (ResNet18)
        input_tensor: Preprocessed input tensor (1, 3, 224, 224)
        original_image: Original PIL Image for overlay
        target_class: Class index to explain (None = predicted class)
        
    Returns:
        tuple: (heatmap_overlay_figure, cam_array)
            - heatmap_overlay_figure: matplotlib Figure with the overlay
            - cam_array: Raw CAM numpy array (for further use)
    """
    model.eval()
    
    # ── Storage for activations and gradients ──
    activations = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        """Capture the activation maps from the target layer."""
        activations["value"] = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        """Capture the gradients flowing back through the target layer."""
        gradients["value"] = grad_output[0].detach()
    
    # ── Register hooks on the last convolutional layer (layer4 for ResNet18) ──
    target_layer = model.layer4[-1]
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # ── Forward pass ──
        output = model(input_tensor)
        
        # If no target class specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # ── Backward pass (for the target class) ──
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)
        
        # ── Compute Grad-CAM ──
        # Global average pooling of gradients → channel weights
        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations["value"]).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU + normalize to [0, 1]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
    finally:
        # ── Always remove hooks to prevent memory leaks ──
        fwd_handle.remove()
        bwd_handle.remove()
    
    # ── Resize CAM to match original image ──
    original_size = original_image.size  # (W, H)
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            original_size, Image.BILINEAR
        )
    ) / 255.0
    
    # ── Create the overlay figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0e1117")
    
    # Panel 1: Original X-ray
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original X-Ray", color="white", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    
    # Panel 2: Grad-CAM heatmap only
    axes[1].imshow(cam_resized, cmap="jet", alpha=0.9)
    axes[1].set_title("Grad-CAM Heatmap", color="white", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    
    # Panel 3: Overlay on original
    axes[2].imshow(original_image, cmap="gray")
    axes[2].imshow(cam_resized, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay", color="white", fontsize=13, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    return fig, cam_resized


def get_severity_from_cam(cam_array, threshold=0.5):
    """
    Estimate the percentage of the lung area affected based on the
    Grad-CAM activation map. This gives a rough visual severity score.
    
    Args:
        cam_array: The Grad-CAM numpy array (normalized 0-1)
        threshold: Activation threshold to consider "affected"
        
    Returns:
        float: Percentage of the image that is highly activated
    """
    high_activation = (cam_array > threshold).sum()
    total_pixels = cam_array.size
    return (high_activation / total_pixels) * 100
