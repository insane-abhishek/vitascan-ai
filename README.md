# 🏥 Antigravity — AI-Powered Health Assistant

> _"Defying gravity – lifting healthcare accessibility with AI."_

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-Educational-green)

---

## ⚠️ Disclaimer

**This is an educational/demonstration project.** It is **NOT** a certified medical device and should not be used for clinical diagnosis. Always consult a licensed healthcare professional.

---

## 📋 Overview

Antigravity is a multi-module AI-powered health screening tool built with Streamlit. It demonstrates real AI impact in healthcare: faster, low-cost, accessible screening for everyone.

### Modules

| # | Module | Technology | Description |
|---|--------|-----------|-------------|
| 1 | 🫁 Pneumonia Detection | ResNet18 (PyTorch) | Upload a chest X-ray → AI classifies Normal/Pneumonia with Grad-CAM heatmap |
| 2 | ❤️ Heart Disease Risk | Random Forest (sklearn) | Enter 13 clinical parameters → Get risk percentage with explainability |
| 3 | ⚖️ BMI & Health Insights | WHO Calculator | Calculate BMI → Get personalized insights + ideal weight range |
| 4 | 🧠 Mental Health | PHQ-4 Questionnaire | 8-question screening → Anxiety/Depression scoring with recommendations |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Clone the repository
git clone <your-repo-url>
cd antigravity

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧠 Training the Models

### Pneumonia Detection Model (ResNet18)

1. **Download the dataset** from Kaggle:  
   [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. **Extract** to `data/chest_xray/` with this structure:
   ```
   data/chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

3. **Run training:**
   ```bash
   python train_pneumonia_model.py --data_dir ./data/chest_xray --epochs 10
   ```

4. Model saved to: `models/pneumonia_resnet18.pth`

> **Expected accuracy:** >90% on test set

### Heart Disease Model (Random Forest)

1. **Download the dataset:**  
   [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) or  
   [Kaggle Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

2. **Place** `heart.csv` in `data/heart.csv`

3. **Run training:**
   ```bash
   python train_heart_model.py
   ```
   (If no CSV is found, the script will attempt to auto-download it.)

4. Model & scaler saved to: `models/heart_rf_model.pkl` and `models/heart_scaler.pkl`

> **Expected accuracy:** >85% with AUC >0.88

---

## 📁 Project Structure

```
antigravity/
├── app.py                      # Main Streamlit application (6 pages)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PRESENTATION.md             # Presentation slide outline
│
├── models/
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing for all models
│   ├── pneumonia_resnet18.pth  # Trained pneumonia model (after training)
│   ├── heart_rf_model.pkl      # Trained heart model (after training)
│   └── heart_scaler.pkl        # Feature scaler (after training)
│
├── utils/
│   ├── __init__.py
│   ├── gradcam.py              # Grad-CAM heatmap visualization
│   ├── pdf_generator.py        # PDF report generation
│   └── insights.py             # BMI + Mental health scoring logic
│
├── data/
│   ├── chest_xray/             # Kaggle X-ray dataset (download separately)
│   └── heart.csv               # UCI Heart Disease dataset
│
├── train_pneumonia_model.py    # Pneumonia model training script
└── train_heart_model.py        # Heart model training script
```

---

## 🎨 Features

- **Modern Dark UI** with medical blue/green gradient theme
- **Grad-CAM Visualization** showing model attention on X-rays
- **Interactive Plotly Charts** (risk gauges, BMI scales, bar charts)
- **One-click PDF Reports** for all modules
- **Demo Mode** — works without training (uses pretrained/demo models)
- **Ethical Disclaimers** on every page
- **Privacy-First** — no data stored, all processing local
- **Loading Animations** (spinners, progress bars)
- **Responsive Layout** (works on mobile)

---

## 🖥️ Demo Video Script

1. Open the app → Show the landing page with 4 module cards
2. Navigate to **Pneumonia Detection** → Upload a sample X-ray → Show prediction + Grad-CAM
3. Navigate to **Heart Disease Risk** → Fill in sample data → Show risk gauge + feature importance
4. Navigate to **BMI & Health Insights** → Enter height/weight → Show BMI scale + insights
5. Navigate to **Mental Health** → Fill questionnaire → Show results + recommendations
6. Download a sample PDF report
7. Show the About page with impact stats

---

## 📊 Model Performance

| Model | Metric | Expected |
|-------|--------|----------|
| Pneumonia ResNet18 | Accuracy | >90% |
| Pneumonia ResNet18 | Sensitivity | >92% |
| Heart Disease RF | Accuracy | >85% |
| Heart Disease RF | AUC-ROC | >0.88 |
| BMI Calculator | Accuracy | 100% (deterministic) |
| PHQ-4 Scoring | Validity | Internationally validated scale |

---

## 🔮 Future Roadmap

- [ ] Skin disease detection (dermatology module)
- [ ] Diabetic retinopathy screening
- [ ] Multi-language support (Hindi, Spanish, French)
- [ ] Deploy to Streamlit Cloud / Hugging Face Spaces
- [ ] Mobile app (via Capacitor)
- [ ] Integration with EHR systems
- [ ] Federated learning for privacy-preserving training

---

## 📄 License

This project is for **educational and demonstration purposes only**. Not licensed for clinical use.

---

## 🤝 Contributing

This is a university project. For questions or contributions, please open an issue or contact the team.

---

<p align="center">
  Built with ❤️ using Streamlit, PyTorch & scikit-learn<br>
  <strong>Antigravity</strong> — Defying gravity, lifting healthcare accessibility with AI.
</p>
