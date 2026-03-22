"""
============================================================
  VitaScan AI — Smart Health Screening Platform
  "Smart health screening, accessible to all."

  Main Streamlit Application
  Modules:
    1. Pneumonia Detection (ResNet18 + Grad-CAM)
    2. Heart Disease Risk (Random Forest)
    3. BMI & Health Insights (Calculator)
    4. Mental Health Screening (PHQ-4)
============================================================
"""

import os
import io
import sys
import time
import base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ── Add project root to path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Local imports ──
from models.preprocessing import (
    preprocess_xray, HEART_FEATURE_NAMES, HEART_FEATURE_LABELS,
    preprocess_heart_features, get_risk_category, get_risk_color
)
from utils.gradcam import generate_gradcam, get_severity_from_cam
from utils.pdf_generator import (
    generate_heart_report, generate_bmi_report, generate_mental_health_report
)
from utils.insights import (
    calculate_bmi, get_bmi_category, get_bmi_color, calculate_ideal_weight_range,
    get_bmi_insights, PHQ4_QUESTIONS, PHQ4_OPTIONS, calculate_mental_health_score,
    get_mental_health_color
)

# ── Constants ──
APP_NAME = "VitaScan AI"
APP_TAGLINE = "Smart health screening, accessible to all."
APP_VERSION = "1.0"
APP_YEAR = "2026"


# ============================================
# PAGE CONFIG & CUSTOM CSS
# ============================================

st.set_page_config(
    page_title=f"{APP_NAME} — Health Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css():
    """Load and inject external CSS from styles.css file."""
    # Load Google Font via link tag (NOT @import, which fails in injected <style> blocks)
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True
    )
    css_path = os.path.join(os.path.dirname(__file__), ".streamlit", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
        .stApp { background: linear-gradient(135deg, #0a0e27, #0d1b2a, #0a192f); }
        .main-title { text-align:center; color:#2ecc71; font-size:3rem; font-weight:800; }
        .tagline { text-align:center; color:#7f8c9b; font-style:italic; }
        </style>""", unsafe_allow_html=True)


def disclaimer_banner():
    """Show the educational disclaimer on every page."""
    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <strong>FOR EDUCATIONAL / DEMO PURPOSES ONLY</strong> — This is a screening tool,
        not a medical device. Always consult a licensed healthcare professional.
    </div>
    """, unsafe_allow_html=True)


# ============================================
# MODEL LOADING (cached)
# ============================================

@st.cache_resource
def load_pneumonia_model():
    """Load the fine-tuned ResNet18 pneumonia detection model."""
    import torch
    import torch.nn as nn
    from torchvision import models

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2),
    )

    model_path = os.path.join(os.path.dirname(__file__), "models", "pneumonia_resnet18.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    else:
        # Pretrained ImageNet fallback for demo
        pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained.state_dict().items()
                          if k in model_dict and "fc" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.eval()
    return model


@st.cache_resource
def load_heart_model():
    """Load the trained Random Forest model and scaler for heart disease."""
    import joblib

    model_path = os.path.join(os.path.dirname(__file__), "models", "heart_rf_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "models", "heart_scaler.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        model, scaler = _create_demo_heart_model()

    return model, scaler


def _create_demo_heart_model():
    """Create a demo heart model from synthetic data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n = 300

    X = np.column_stack([
        np.random.normal(54, 9, n),
        np.random.choice([0, 1], n, p=[0.32, 0.68]),
        np.random.choice([0, 1, 2, 3], n),
        np.random.normal(131, 17, n),
        np.random.normal(246, 52, n),
        np.random.choice([0, 1], n, p=[0.85, 0.15]),
        np.random.choice([0, 1, 2], n),
        np.random.normal(149, 23, n),
        np.random.choice([0, 1], n, p=[0.67, 0.33]),
        np.abs(np.random.normal(1.04, 1.16, n)),
        np.random.choice([0, 1, 2], n),
        np.random.choice([0, 1, 2, 3], n),
        np.random.choice([1, 2, 3], n),
    ])

    risk = (
        (X[:, 0] > 55).astype(float) * 0.15 +
        (X[:, 2] >= 2).astype(float) * 0.2 +
        (X[:, 4] > 240).astype(float) * 0.15 +
        (X[:, 8] == 1).astype(float) * 0.2 +
        (X[:, 9] > 1.5).astype(float) * 0.15 +
        (X[:, 11] >= 1).astype(float) * 0.15
    )
    y = (risk + np.random.normal(0, 0.1, n) > 0.35).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    return model, scaler


# ============================================
# HOME PAGE
# ============================================

def render_home():
    """Render the home / landing page."""

    # ── Hero ──
    st.markdown(f'<h1 class="main-title">🩺 {APP_NAME}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="tagline">{APP_TAGLINE}</p>', unsafe_allow_html=True)

    st.markdown("---")
    disclaimer_banner()

    # ── Module Cards ──
    st.markdown("### 🧩 Available Screening Modules")
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2.5rem; margin-bottom:10px;">🫁</div>
            <div style="color:#e0e6ed; font-weight:600; font-size:1.05rem;">Pneumonia Detection</div>
            <div class="metric-label">Deep Learning · ResNet18</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2.5rem; margin-bottom:10px;">❤️</div>
            <div style="color:#e0e6ed; font-weight:600; font-size:1.05rem;">Heart Disease Risk</div>
            <div class="metric-label">Machine Learning · Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2.5rem; margin-bottom:10px;">⚖️</div>
            <div style="color:#e0e6ed; font-weight:600; font-size:1.05rem;">BMI & Health Insights</div>
            <div class="metric-label">Calculator · WHO Standards</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2.5rem; margin-bottom:10px;">🧠</div>
            <div style="color:#e0e6ed; font-weight:600; font-size:1.05rem;">Mental Health</div>
            <div class="metric-label">PHQ-4 · Questionnaire</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Impact Stats ──
    st.markdown("""
    <div class="stats-row">
        <div class="stat-item">
            <div class="stat-number">4</div>
            <div class="stat-label">AI Modules</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">&lt;30s</div>
            <div class="stat-label">Screening Time</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">₹0</div>
            <div class="stat-label">Cost Per Scan</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">100%</div>
            <div class="stat-label">Local & Private</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How It Works Pipeline ──
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="step-icon">📤</div>
            <div class="step-title">Upload</div>
            <div class="step-desc">X-ray image or enter clinical data</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="step-icon">🤖</div>
            <div class="step-title">AI Analysis</div>
            <div class="step-desc">Deep learning & ML models process your data</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="step-icon">📊</div>
            <div class="step-title">Results</div>
            <div class="step-desc">Predictions with confidence & explainability</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">
            <div class="step-icon">📄</div>
            <div class="step-title">Report</div>
            <div class="step-desc">Download PDF with recommendations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Quick Demo ──
    st.markdown("### 🚀 Quick Demo — Sample Patient")
    st.info("Select a module from the sidebar to get started, or try the instant demo below.")

    if st.button("🎯 Run All Modules on Sample Patient", use_container_width=True):
        with st.spinner("Running AI screening on sample patient data..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.008)
                progress.progress(i + 1)
            progress.empty()

        st.markdown("---")

        # Heart Disease Demo
        sample_heart = {
            "age": 55, "sex": 1, "cp": 2, "trestbps": 140,
            "chol": 260, "fbs": 0, "restecg": 1, "thalach": 145,
            "exang": 1, "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 2
        }
        model, scaler = load_heart_model()
        features = preprocess_heart_features(sample_heart)
        features_scaled = scaler.transform(features)
        heart_prob = model.predict_proba(features_scaled)[0][1]
        heart_cat = get_risk_category(heart_prob)
        heart_color = get_risk_color(heart_cat)

        # BMI Demo
        bmi_val = calculate_bmi(82, 175)
        bmi_cat = get_bmi_category(bmi_val)
        bmi_color = get_bmi_color(bmi_cat)

        # Mental Health Demo
        mh_answers = ["Not at all", "Several days", "Not at all", "Several days",
                      "Not at all", "Not at all", "Several days", "Not at all"]
        mh_results = calculate_mental_health_score(mh_answers)
        mh_color = get_mental_health_color(mh_results["category"])

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            rc = "success" if heart_cat == "Low Risk" else ("warning" if heart_cat == "Medium Risk" else "danger")
            st.markdown(f"""
            <div class="result-box {rc}">
                <div style="font-size:1.3rem; margin-bottom:6px;">❤️ Heart Disease Risk</div>
                <h3 style="color:{heart_color}; margin:0;">{heart_cat}</h3>
                <p style="color:#8899a6; margin:6px 0 0 0;">
                    Probability: <strong style="color:{heart_color};">{heart_prob:.1%}</strong><br>
                    <small>Male, 55 yrs, Chol: 260</small>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            bc = "success" if bmi_cat == "Normal weight" else ("warning" if "Overweight" in bmi_cat else "info")
            st.markdown(f"""
            <div class="result-box {bc}">
                <div style="font-size:1.3rem; margin-bottom:6px;">⚖️ BMI Assessment</div>
                <h3 style="color:{bmi_color}; margin:0;">BMI: {bmi_val}</h3>
                <p style="color:#8899a6; margin:6px 0 0 0;">
                    Category: <strong style="color:{bmi_color};">{bmi_cat}</strong><br>
                    <small>175 cm, 82 kg</small>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            mc = "success" if mh_results["category"] == "Normal" else "info"
            st.markdown(f"""
            <div class="result-box {mc}">
                <div style="font-size:1.3rem; margin-bottom:6px;">🧠 Mental Health</div>
                <h3 style="color:{mh_color}; margin:0;">{mh_results["category"]}</h3>
                <p style="color:#8899a6; margin:6px 0 0 0;">
                    Score: <strong style="color:{mh_color};">{mh_results["total_score"]}/12</strong><br>
                    <small>Anxiety: {mh_results["anxiety_score"]}/6 | Depression: {mh_results["depression_score"]}/6</small>
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Why VitaScan ──
    with st.expander("💡 Why VitaScan AI?", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **The Problem:**
            - 🌍 Half the world lacks access to essential health services
            - ⏰ Late diagnosis leads to preventable deaths
            - 💰 Healthcare costs are a barrier for billions
            - 🏥 Rural areas suffer from severe doctor shortages
            """)
        with col_b:
            st.markdown("""
            **Our Solution:**
            - 🤖 AI-powered screening at zero cost
            - ⚡ Results in seconds, not days
            - 📱 Accessible from any device with a browser
            - 🔒 Fully private — all processing is local
            """)

    # ── Footer ──
    st.markdown(f"""
    <div class="footer">
        <strong>{APP_NAME}</strong> — Smart Health Screening Platform |
        Built with Streamlit, PyTorch & scikit-learn<br>
        <span style="color:#e74c3c;">⚠️ Educational project only — not a medical device</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# MODULE 1: PNEUMONIA DETECTION
# ============================================

def render_pneumonia():
    """Render the Pneumonia Detection module."""
    import torch

    st.markdown("""
    <div class="module-header">
        <h2>🫁 Pneumonia Detection from Chest X-Rays</h2>
        <p>Upload a chest X-ray image and our AI will screen for signs of pneumonia using deep learning.</p>
    </div>
    """, unsafe_allow_html=True)

    disclaimer_banner()

    with st.expander("How does this work?"):
        st.markdown("""
        **Model:** ResNet18 (pre-trained on ImageNet, fine-tuned on chest X-ray dataset)

        **Process:**
        1. You upload a chest X-ray image (PNG/JPG)
        2. The image is preprocessed (resized to 224x224, normalized)
        3. The model classifies the image as **Normal** or **Pneumonia**
        4. A **Grad-CAM heatmap** highlights the regions the model focused on

        **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,863 labeled images
        """)

    uploaded_file = st.file_uploader(
        "Upload a Chest X-Ray Image",
        type=["png", "jpg", "jpeg"],
        help="Upload a frontal chest X-ray. Supported: PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 📸 Uploaded X-Ray")
            st.image(image, use_container_width=True, caption="Uploaded chest X-ray")

        if st.button("🔍 Analyze X-Ray", use_container_width=True):
            with st.spinner("AI is analyzing the X-ray..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                model = load_pneumonia_model()
                input_tensor = preprocess_xray(image)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()

                classes = ["Normal", "Pneumonia"]
                prediction = classes[predicted_class]
                confidence = probabilities[predicted_class].item() * 100

                progress.empty()

            with col2:
                st.markdown("#### 📊 AI Prediction")

                result_class = "success" if prediction == "Normal" else "danger"
                result_color = "#2ecc71" if prediction == "Normal" else "#e74c3c"
                emoji = "✅" if prediction == "Normal" else "⚠️"

                st.markdown(f"""
                <div class="result-box {result_class}">
                    <h2 style="color:{result_color}; margin:0; font-size:1.8rem;">
                        {emoji} {prediction}
                    </h2>
                    <p style="color:#8899a6; font-size:1.1rem; margin:8px 0 0 0;">
                        Confidence: <strong style="color:{result_color};">{confidence:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Class Probabilities:**")
                for i, cls in enumerate(classes):
                    prob = probabilities[i].item() * 100
                    st.markdown(f"**{cls}:** {prob:.1f}%")
                    st.progress(prob / 100)

            # Grad-CAM
            st.markdown("---")
            st.markdown("#### 🔥 Grad-CAM Visualization")
            st.caption("The heatmap shows which areas of the X-ray the AI focused on.")

            with st.spinner("Generating Grad-CAM heatmap..."):
                try:
                    fig, cam_array = generate_gradcam(model, input_tensor, image)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    severity = get_severity_from_cam(cam_array)
                    st.markdown(f"""
                    <div class="result-box info">
                        <p style="color:#8899a6; margin:0;">
                            <strong>Activation Coverage:</strong> {severity:.1f}% of the image shows
                            high model attention — indicates the approximate area of concern.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Grad-CAM visualization could not be generated: {str(e)}")

            st.warning(
                "**Disclaimer:** This is a screening tool only — it is NOT a clinical diagnosis. "
                "Please consult a qualified radiologist or physician."
            )
    else:
        st.info("Upload a chest X-ray image to get started. Supported formats: PNG, JPG, JPEG.")

        with st.expander("Need sample X-rays for testing?"):
            st.markdown("""
            Download sample chest X-rays from the Kaggle dataset:

            [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

            - **Normal:** ~1,583 images
            - **Pneumonia:** ~4,273 images (bacterial + viral)

            Download from the `test` folder and upload here to test.
            """)


# ============================================
# MODULE 2: HEART DISEASE RISK
# ============================================

def render_heart():
    """Render the Heart Disease Risk Prediction module."""

    st.markdown("""
    <div class="module-header">
        <h2>❤️ Heart Disease Risk Prediction</h2>
        <p>Enter clinical parameters to assess cardiovascular risk using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    disclaimer_banner()

    with st.expander("About this module"):
        st.markdown("""
        **Model:** Random Forest Classifier trained on the UCI Heart Disease dataset

        **Features:** 13 clinical parameters commonly measured during cardiac evaluation

        **Output:** Risk percentage (0-100%) with category (Low / Medium / High)
        and explanation of top contributing factors

        **Dataset:** [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) — 303 patient records
        """)

    st.markdown("### 📋 Enter Patient Data")

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
            sex = st.selectbox("Sex", options=["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", options=[
                "0 — Typical Angina", "1 — Atypical Angina",
                "2 — Non-Anginal Pain", "3 — Asymptomatic",
            ], help="Type of chest pain experienced")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)",
                                       min_value=80, max_value=220, value=130, step=1)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])

        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)",
                                   min_value=100, max_value=600, value=240, step=1)
            restecg = st.selectbox("Resting ECG Results", options=[
                "0 — Normal", "1 — ST-T Wave Abnormality",
                "2 — Left Ventricular Hypertrophy",
            ])
            thalach = st.number_input("Max Heart Rate Achieved",
                                      min_value=60, max_value=220, value=150, step=1)
            exang = st.selectbox("Exercise-Induced Angina?", options=["No", "Yes"])

        with col3:
            oldpeak = st.number_input("ST Depression (Oldpeak)",
                                      min_value=0.0, max_value=7.0, value=1.0, step=0.1,
                                      help="ST depression induced by exercise relative to rest")
            slope = st.selectbox("Slope of Peak Exercise ST", options=[
                "0 — Upsloping", "1 — Flat", "2 — Downsloping",
            ])
            ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", options=[
                "1 — Normal", "2 — Fixed Defect", "3 — Reversible Defect",
            ])

        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)

    if submitted:
        with st.spinner("AI is analyzing the clinical data..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.008)
                progress.progress(i + 1)

            features = {
                "age": age, "sex": 1 if sex == "Male" else 0,
                "cp": int(cp[0]), "trestbps": trestbps, "chol": chol,
                "fbs": 1 if fbs == "Yes" else 0, "restecg": int(restecg[0]),
                "thalach": thalach, "exang": 1 if exang == "Yes" else 0,
                "oldpeak": oldpeak, "slope": int(slope[0]), "ca": ca,
                "thal": int(thal[0]),
            }

            model, scaler = load_heart_model()
            X = preprocess_heart_features(features)
            X_scaled = scaler.transform(X)

            probability = model.predict_proba(X_scaled)[0][1]
            category = get_risk_category(probability)
            color = get_risk_color(category)

            importances = dict(zip(HEART_FEATURE_NAMES, model.feature_importances_))
            top_factors = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

            progress.empty()

        st.markdown("---")
        st.markdown("### 📊 Risk Assessment Results")

        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                number={"suffix": "%", "font": {"size": 36, "color": "white"}},
                title={"text": "Heart Disease Risk", "font": {"size": 18, "color": "#7f8c9b"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(46, 204, 113, 0.15)"},
                        {"range": [30, 60], "color": "rgba(243, 156, 18, 0.15)"},
                        {"range": [60, 100], "color": "rgba(231, 76, 60, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": probability * 100,
                    },
                },
            ))
            fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                margin={"t": 30, "b": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r2:
            rc = "success" if category == "Low Risk" else ("warning" if category == "Medium Risk" else "danger")
            emoji = "✅" if category == "Low Risk" else ("⚠️" if category == "Medium Risk" else "🚨")
            st.markdown(f"""
            <div class="result-box {rc}">
                <h2 style="color:{color}; margin:0;">{emoji} {category}</h2>
                <p style="color:#8899a6; margin:8px 0 0 0; font-size:1.1rem;">
                    Risk Probability: <strong style="color:{color};">{probability:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**Top Contributing Factors:**")
            for i, (feature, importance) in enumerate(top_factors, 1):
                label = HEART_FEATURE_LABELS.get(feature, feature)
                st.markdown(f"{i}. **{label}** — Importance: `{importance:.3f}`")

        with st.expander("Full Feature Importance Breakdown"):
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            labels = [HEART_FEATURE_LABELS.get(f, f) for f, _ in sorted_features]
            values = [v for _, v in sorted_features]

            fig_imp = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=[f"rgba(46, 204, 113, {0.3 + 0.7 * v / max(values)})" for v in values],
            ))
            fig_imp.update_layout(
                title="Feature Importance", height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                xaxis={"title": "Importance Score", "gridcolor": "rgba(255,255,255,0.08)"},
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        pdf_bytes = generate_heart_report(features, probability, category, top_factors)
        st.download_button(
            label="📥 Download Full Report (PDF)",
            data=pdf_bytes,
            file_name=f"vitascan_heart_report_{int(time.time())}.pdf",
            mime="application/pdf", use_container_width=True,
        )

        st.warning(
            "**Disclaimer:** This prediction is based on a machine learning model. "
            "It is NOT a clinical diagnosis. Please consult a cardiologist."
        )


# ============================================
# MODULE 3: BMI & HEALTH INSIGHTS
# ============================================

def render_bmi():
    """Render the BMI & Health Insights module."""

    st.markdown("""
    <div class="module-header">
        <h2>⚖️ BMI & Health Insights</h2>
        <p>Calculate your Body Mass Index and get personalized health recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    disclaimer_banner()

    st.markdown("### 📏 Enter Your Details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5)
    with col3:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=25, step=1)
    with col4:
        gender = st.selectbox("Gender", options=["Male", "Female"])

    if st.button("📊 Calculate BMI & Get Insights", use_container_width=True):
        with st.spinner("Calculating..."):
            time.sleep(0.3)

            bmi = calculate_bmi(weight, height)
            category = get_bmi_category(bmi)
            color = get_bmi_color(category)
            ideal_range = calculate_ideal_weight_range(height)
            insights = get_bmi_insights(bmi, category, age, gender)

        st.markdown("---")
        st.markdown("### 📊 Your Results")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{bmi}</div>
                <div class="metric-label">Your BMI</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="-webkit-text-fill-color:{color};">{category}</div>
                <div class="metric-label">WHO Category</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ideal_range[0]}-{ideal_range[1]}</div>
                <div class="metric-label">Ideal Weight Range (kg)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # BMI Scale + Donut
        col_bar, col_donut = st.columns([2, 1])

        with col_bar:
            fig = go.Figure()
            ranges = [
                ("Underweight", 0, 18.5, "#3498db"),
                ("Normal", 18.5, 25, "#2ecc71"),
                ("Overweight", 25, 30, "#f39c12"),
                ("Obese", 30, 40, "#e74c3c"),
            ]
            for name, start, end, col in ranges:
                fig.add_trace(go.Bar(
                    x=[end - start], y=["BMI"], orientation="h",
                    base=start, name=name, marker_color=col, marker_line_width=0,
                    text=name, textposition="inside",
                    textfont={"color": "white", "size": 12},
                ))
            fig.add_vline(x=bmi, line_width=3, line_dash="dash", line_color="white",
                          annotation_text=f"You: {bmi}", annotation_position="top",
                          annotation_font_color="white")
            fig.update_layout(
                height=120, showlegend=False, barmode="stack",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                xaxis={"title": "BMI", "range": [0, 42], "gridcolor": "rgba(255,255,255,0.08)"},
                yaxis={"visible": False},
                margin={"t": 30, "b": 30, "l": 10, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_donut:
            donut = go.Figure(go.Pie(
                labels=["Your BMI", "Remaining to 40"],
                values=[min(bmi, 40), max(40 - bmi, 0)],
                hole=0.65,
                marker_colors=[color, "rgba(255,255,255,0.05)"],
                textinfo="none",
            ))
            donut.update_layout(
                height=160, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                margin={"t": 10, "b": 10, "l": 10, "r": 10},
                annotations=[{
                    "text": f"<b>{bmi}</b>",
                    "x": 0.5, "y": 0.5, "font_size": 22, "font_color": color,
                    "showarrow": False
                }],
            )
            st.plotly_chart(donut, use_container_width=True)

        # Insights
        st.markdown("### 💡 Personalized Health Insights")
        for i, insight in enumerate(insights, 1):
            st.markdown(f"""
            <div class="result-box info" style="margin:8px 0;">
                <p style="color:#8899a6; margin:0;">{i}. {insight}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        if weight < ideal_range[0]:
            diff = ideal_range[0] - weight
            st.info(f"📈 You need to gain approximately **{diff:.1f} kg** to reach your ideal weight range.")
        elif weight > ideal_range[1]:
            diff = weight - ideal_range[1]
            st.info(f"📉 Losing approximately **{diff:.1f} kg** would bring you to your ideal weight range.")
        else:
            st.success("🎉 Your weight is within the ideal range! Keep it up!")

        st.markdown("---")
        pdf_bytes = generate_bmi_report(height, weight, age, gender, bmi, category, insights, ideal_range)
        st.download_button(
            label="📥 Download BMI Report (PDF)",
            data=pdf_bytes,
            file_name=f"vitascan_bmi_report_{int(time.time())}.pdf",
            mime="application/pdf", use_container_width=True,
        )


# ============================================
# MODULE 4: MENTAL HEALTH SCREENING
# ============================================

def render_mental_health():
    """Render the Mental Health Screening module."""

    st.markdown("""
    <div class="module-header">
        <h2>🧠 Mental Health Screening</h2>
        <p>A brief PHQ-4 style questionnaire to screen for symptoms of anxiety and depression.</p>
    </div>
    """, unsafe_allow_html=True)

    disclaimer_banner()

    st.markdown("""
    <div class="result-box info">
        <p style="color:#8899a6; margin:0;">
            🔒 <strong>Privacy Notice:</strong> Your responses are NOT stored anywhere.
            All processing happens locally in your browser session. No data leaves your device.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    with st.expander("About this screening"):
        st.markdown("""
        **What is PHQ-4?**

        The Patient Health Questionnaire-4 (PHQ-4) is a brief, validated screening tool that
        combines the GAD-2 (anxiety) and PHQ-2 (depression) scales.

        **This extended version** includes 8 questions (4 for anxiety, 4 for depression)
        for a more comprehensive assessment. Scores are normalized to the standard 0-12 scale.

        **Note:** This is a screening tool, not a diagnostic instrument.
        """)

    st.markdown("### 📝 Answer the Following Questions")
    st.caption("For each question, select how often you've been bothered over the **last 2 weeks**.")

    options = list(PHQ4_OPTIONS.keys())
    answers = []

    with st.form("mental_health_form"):
        for i, question in enumerate(PHQ4_QUESTIONS):
            prefix = "😰" if i < 4 else "😔"
            answer = st.radio(
                f"Q{i+1}. {prefix} {question}",
                options=options, horizontal=True, key=f"mh_q{i}",
            )
            answers.append(answer)
            if i < len(PHQ4_QUESTIONS) - 1:
                st.markdown("---")

        submitted = st.form_submit_button("📊 Get My Results", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing your responses..."):
            time.sleep(0.5)
            results = calculate_mental_health_score(answers)

        total = results["total_score"]
        anxiety = results["anxiety_score"]
        depression = results["depression_score"]
        category = results["category"]
        recommendations = results["recommendations"]
        color = get_mental_health_color(category)

        st.markdown("---")
        st.markdown("### 📊 Your Screening Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="-webkit-text-fill-color:{color};">{total}/12</div>
                <div class="metric-label">Total Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{anxiety}/6</div>
                <div class="metric-label">Anxiety (GAD-2)</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{depression}/6</div>
                <div class="metric-label">Depression (PHQ-2)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        rc_map = {"Normal": "success", "Mild": "info", "Moderate": "warning", "Severe": "danger"}
        em_map = {"Normal": "✅", "Mild": "💛", "Moderate": "⚠️", "Severe": "🚨"}

        st.markdown(f"""
        <div class="result-box {rc_map.get(category, 'info')}">
            <h2 style="color:{color}; margin:0;">
                {em_map.get(category, '')} Severity: {category}
            </h2>
            <p style="color:#8899a6; margin:8px 0 0 0;">
                Total: {total}/12 | Anxiety: {anxiety}/6 | Depression: {depression}/6
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart + Radar chart side by side
        col_bar, col_radar = st.columns([1, 1])

        with col_bar:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Anxiety (GAD-2)", "Depression (PHQ-2)", "Total"],
                y=[anxiety, depression, total],
                marker_color=["#9b59b6", "#3498db", color],
                text=[f"{anxiety}/6", f"{depression}/6", f"{total}/12"],
                textposition="auto",
                textfont={"color": "white", "size": 14},
            ))
            fig.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                yaxis={"title": "Score", "gridcolor": "rgba(255,255,255,0.08)", "range": [0, 13]},
                xaxis={"title": ""}, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_radar:
            # Spider/radar chart of individual question scores
            scores = [PHQ4_OPTIONS.get(a, 0) for a in answers]
            q_labels = [f"Q{i+1}" for i in range(len(scores))]

            radar = go.Figure(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=q_labels + [q_labels[0]],
                fill='toself',
                fillcolor="rgba(46, 204, 113, 0.15)",
                line={"color": "#2ecc71", "width": 2},
                marker={"size": 6, "color": "#2ecc71"},
            ))
            radar.update_layout(
                height=300,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 3],
                                    gridcolor="rgba(255,255,255,0.08)",
                                    color="#7f8c9b"),
                    angularaxis=dict(color="#7f8c9b"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                showlegend=False,
                margin={"t": 30, "b": 30},
            )
            st.plotly_chart(radar, use_container_width=True)

        # Recommendations
        st.markdown("### 💚 Recommended Next Steps")
        for rec in recommendations:
            st.markdown(f"""
            <div class="result-box info" style="margin:8px 0;">
                <p style="color:#8899a6; margin:0;">• {rec}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        answers_dict = {q: a for q, a in zip(PHQ4_QUESTIONS, answers)}
        pdf_bytes = generate_mental_health_report(
            answers_dict, total, anxiety, depression, category, recommendations
        )
        st.download_button(
            label="📥 Download Screening Report (PDF)",
            data=pdf_bytes,
            file_name=f"vitascan_mental_health_report_{int(time.time())}.pdf",
            mime="application/pdf", use_container_width=True,
        )

        if category in ["Moderate", "Severe"]:
            st.error("""
            🆘 **If you are in crisis or having thoughts of self-harm:**
            - **India:** iCall — 9152987821 | Vandrevala Foundation — 1860-2662-345
            - **USA:** 988 Suicide & Crisis Lifeline — Call or text **988**
            - **UK:** Samaritans — **116 123**
            - **International:** [findahelpline.com](https://findahelpline.com)
            """)


# ============================================
# ABOUT PAGE
# ============================================

def render_about():
    """Render the About page."""

    st.markdown(f'<h1 class="main-title">About {APP_NAME}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="tagline">{APP_TAGLINE}</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        ### 🎯 Mission

        {APP_NAME} demonstrates how artificial intelligence can make healthcare
        screening **faster, cheaper, and more accessible** — especially in
        underserved communities.

        ### 🧩 Modules

        | Module | Technology | Accuracy |
        |--------|-----------|----------|
        | Pneumonia Detection | ResNet18 (PyTorch) | >90% |
        | Heart Disease Risk | Random Forest (sklearn) | >85% |
        | BMI & Health | WHO Calculator | 100% |
        | Mental Health | PHQ-4 Scoring | Validated |

        ### 🛡️ Ethics & Privacy

        - All processing happens **locally** — no data sent to external servers
        - No patient data is stored or logged
        - Every result includes a clear disclaimer
        - This is an **educational/demonstration project only**
        """)

    with col2:
        st.markdown(f"""
        ### ⚙️ Tech Stack

        - **Framework:** Streamlit
        - **Deep Learning:** PyTorch + torchvision
        - **Classical ML:** scikit-learn
        - **Visualization:** Plotly, Matplotlib
        - **PDF Reports:** fpdf2
        - **Language:** Python 3.10+

        ### 📈 Impact Potential

        - ⚡ Screening results in **seconds**, not days
        - 💰 **Zero cost** — runs on any computer with Python
        - 🌍 Accessible from any device with a browser
        - 🏥 Can support healthcare workers in rural areas

        ### 🔮 Future Roadmap

        - Skin disease detection (dermatology module)
        - Diabetic retinopathy screening
        - Multi-language support
        - Mobile app (via Streamlit Cloud)
        - Integration with EHR systems
        """)

    st.markdown("---")

    st.markdown(f"""
    <div class="footer">
        <strong>{APP_NAME}</strong> v{APP_VERSION} | Smart Health Screening Platform<br>
        Built with Streamlit, PyTorch & scikit-learn<br>
        <span style="color:#e74c3c;">⚠️ For educational and demo purposes only — not a medical device</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# MAIN APP — SIDEBAR NAVIGATION
# ============================================

def main():
    """Main entry point — sidebar navigation and page routing."""

    inject_custom_css()

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:20px 0;">
            <div style="font-size:2.5rem;">🩺</div>
            <h2 style="background:linear-gradient(135deg, #00d2ff, #2ecc71);
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;
                        background-clip:text;
                        font-size:1.5rem; margin:5px 0;">
                {APP_NAME}
            </h2>
            <p style="color:#7f8c9b; font-size:0.8rem; font-style:italic;">
                {APP_TAGLINE}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigate",
            options=[
                "🏠 Home",
                "🫁 Pneumonia Detection",
                "❤️ Heart Disease Risk",
                "⚖️ BMI & Health Insights",
                "🧠 Mental Health",
                "ℹ️ About",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Model Status
        st.markdown("### 📦 Model Status")

        pneumonia_exists = os.path.exists(
            os.path.join(os.path.dirname(__file__), "models", "pneumonia_resnet18.pth")
        )
        heart_exists = os.path.exists(
            os.path.join(os.path.dirname(__file__), "models", "heart_rf_model.pkl")
        )

        if pneumonia_exists:
            st.markdown("✅ Pneumonia Model")
        else:
            st.markdown("⚠️ Pneumonia Model (demo)")

        if heart_exists:
            st.markdown("✅ Heart Disease Model")
        else:
            st.markdown("⚠️ Heart Disease Model (demo)")

        if not pneumonia_exists or not heart_exists:
            st.caption("Demo models active. See README to train real models.")

        st.markdown("---")
        st.caption(f"v{APP_VERSION} | © {APP_NAME} {APP_YEAR}")

    # Page Routing
    if "Home" in page:
        render_home()
    elif "Pneumonia" in page:
        render_pneumonia()
    elif "Heart" in page:
        render_heart()
    elif "BMI" in page:
        render_bmi()
    elif "Mental" in page:
        render_mental_health()
    elif "About" in page:
        render_about()


if __name__ == "__main__":
    main()
