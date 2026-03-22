"""
============================================
Antigravity - Heart Disease Model Training Script
============================================
Trains a Random Forest Classifier on the UCI Heart Disease
dataset for binary classification: heart disease risk prediction.

Dataset: UCI Heart Disease Dataset
  https://archive.ics.uci.edu/dataset/45/heart+disease
  Also available at: data/heart.csv

Usage:
  python train_heart_model.py

The trained model + scaler are saved to:
  models/heart_rf_model.pkl
  models/heart_scaler.pkl
============================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)


# Feature names matching UCI Heart Disease dataset
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

TARGET_NAME = "target"


def load_data(filepath="./data/heart.csv"):
    """
    Load the UCI Heart Disease dataset from CSV.
    
    Expected CSV columns: age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak, slope, ca, thal, target
    
    Args:
        filepath: Path to heart.csv
        
    Returns:
        tuple: (X DataFrame, y Series)
    """
    print(f"Loading dataset from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"\n⚠ Dataset not found at {filepath}")
        print("Downloading from UCI repository...")
        download_heart_dataset(filepath)
    
    df = pd.read_csv(filepath)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Target distribution:\n{df[TARGET_NAME].value_counts()}\n")
    
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    return X, y


def download_heart_dataset(save_path):
    """
    Download the heart disease dataset from a public source.
    """
    import requests
    
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"  ✓ Dataset downloaded to {save_path}")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print("  Please manually download heart.csv and place it in ./data/")
        raise


def train_model(X, y, n_estimators=200, random_state=42):
    """
    Train a Random Forest Classifier with cross-validation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (trained_model, scaler, feature_importances, X_test, y_test, y_pred, y_prob)
    """
    # ── Split data ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # ── Scale features ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ── Train Random Forest ──
    print(f"\nTraining Random Forest ({n_estimators} trees)...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    
    # ── Cross-validation ──
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # ── Test predictions ──
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # ── Feature importances ──
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
    
    return model, scaler, importances, X_test_scaled, y_test, y_pred, y_prob


def evaluate_model(y_test, y_pred, y_prob):
    """Print detailed evaluation metrics."""
    print(f"\n{'='*50}")
    print("MODEL EVALUATION")
    print(f"{'='*50}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Disease", "Heart Disease"]))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {auc:.4f}")
    
    return auc


def plot_results(y_test, y_prob, importances, save_dir="./models"):
    """Plot ROC curve and feature importance chart."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax1.plot([0, 1], [0, 1], "r--", alpha=0.5)
    ax1.set_title("ROC Curve", fontsize=13)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ── Feature Importance ──
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_features)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    ax2.barh(range(len(names)), values, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    ax2.set_title("Feature Importance", fontsize=13)
    ax2.set_xlabel("Importance Score")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "heart_model_evaluation.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nEvaluation plots saved to: {save_path}")


def main():
    # ── Load data ──
    X, y = load_data("./data/heart.csv")
    
    # ── Train ──
    model, scaler, importances, X_test, y_test, y_pred, y_prob = train_model(X, y)
    
    # ── Evaluate ──
    auc = evaluate_model(y_test, y_pred, y_prob)
    
    # ── Plot ──
    plot_results(y_test, y_prob, importances)
    
    # ── Save model and scaler ──
    os.makedirs("./models", exist_ok=True)
    
    model_path = "./models/heart_rf_model.pkl"
    scaler_path = "./models/heart_scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n{'='*50}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"{'='*50}")
    
    # ── Print top features for reference ──
    print("\nTop 5 Most Important Features:")
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_imp[:5], 1):
        print(f"  {i}. {feature}: {score:.4f}")


if __name__ == "__main__":
    main()
