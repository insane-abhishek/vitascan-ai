"""
============================================
Antigravity - Preprocessing Module
============================================
Handles data preprocessing for all models:
  - Image transforms for Pneumonia X-ray model
  - Feature scaling/encoding for Heart Disease model
============================================
"""

import numpy as np
from PIL import Image

# ── PyTorch imports (for pneumonia preprocessing) ──
import torch
from torchvision import transforms

# ── Scikit-learn imports (for heart disease preprocessing) ──
from sklearn.preprocessing import StandardScaler


# ============================================
# 1. PNEUMONIA X-RAY IMAGE PREPROCESSING
# ============================================

def get_pneumonia_transforms():
    """
    Returns the image transformation pipeline for the pneumonia
    ResNet18 model. Images are resized to 224x224, converted to
    tensors, and normalized with ImageNet statistics.
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),          # ResNet expects 224x224
        transforms.ToTensor(),                  # Convert PIL → Tensor [0,1]
        transforms.Normalize(                   # ImageNet normalization
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def preprocess_xray(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a single chest X-ray image for the pneumonia model.
    
    Args:
        image: PIL Image of the chest X-ray
        
    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension
                      Shape: (1, 3, 224, 224)
    """
    # Convert grayscale to RGB if needed (ResNet expects 3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply the transform pipeline
    transform = get_pneumonia_transforms()
    tensor = transform(image)
    
    # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    return tensor.unsqueeze(0)


# ============================================
# 2. HEART DISEASE FEATURE PREPROCESSING
# ============================================

# UCI Heart Disease dataset feature names (in order)
HEART_FEATURE_NAMES = [
    "age",              # Age in years
    "sex",              # 1 = male, 0 = female
    "cp",               # Chest pain type (0-3)
    "trestbps",         # Resting blood pressure (mm Hg)
    "chol",             # Serum cholesterol (mg/dl)
    "fbs",              # Fasting blood sugar > 120 mg/dl (1=true, 0=false)
    "restecg",          # Resting ECG results (0-2)
    "thalach",          # Maximum heart rate achieved
    "exang",            # Exercise-induced angina (1=yes, 0=no)
    "oldpeak",          # ST depression induced by exercise
    "slope",            # Slope of peak exercise ST segment (0-2)
    "ca",               # Number of major vessels colored by fluoroscopy (0-3)
    "thal",             # Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)
]


# Readable labels for each feature (used in explanations)
HEART_FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "ST Slope",
    "ca": "Major Vessels (CA)",
    "thal": "Thalassemia",
}


def preprocess_heart_features(features_dict: dict) -> np.ndarray:
    """
    Convert a dictionary of heart disease features into a numpy array
    in the correct order for model prediction.
    
    Args:
        features_dict: Dictionary mapping feature names to values
        
    Returns:
        np.ndarray: Feature array of shape (1, 13)
    """
    feature_values = []
    for name in HEART_FEATURE_NAMES:
        feature_values.append(float(features_dict.get(name, 0)))
    
    return np.array(feature_values).reshape(1, -1)


def get_risk_category(probability: float) -> str:
    """
    Convert a heart disease probability (0-1) to a risk category.
    
    Args:
        probability: Model output probability (0.0 to 1.0)
        
    Returns:
        str: "Low Risk", "Medium Risk", or "High Risk"
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


def get_risk_color(category: str) -> str:
    """
    Return a hex color associated with the risk level.
    
    Args:
        category: "Low Risk", "Medium Risk", or "High Risk"
        
    Returns:
        str: Hex color string
    """
    colors = {
        "Low Risk": "#2ecc71",      # Green
        "Medium Risk": "#f39c12",   # Orange
        "High Risk": "#e74c3c",     # Red
    }
    return colors.get(category, "#95a5a6")
