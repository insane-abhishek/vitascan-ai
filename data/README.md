# Data Directory
This directory contains the datasets used for training:

## 1. Chest X-Ray Dataset (Pneumonia Detection)
- **Source:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Size:** ~1.2 GB
- **Structure:**
  ```
  chest_xray/
  ├── train/
  │   ├── NORMAL/      (1,341 images)
  │   └── PNEUMONIA/   (3,875 images)
  ├── val/
  │   ├── NORMAL/      (8 images)
  │   └── PNEUMONIA/   (8 images)
  └── test/
      ├── NORMAL/      (234 images)
      └── PNEUMONIA/   (390 images)
  ```

## 2. Heart Disease Dataset
- **Source:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Alternative:** https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **File:** `heart.csv` (303 records, 14 columns)
- **Size:** ~12 KB

## Download Instructions

### Option A: Kaggle CLI
```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data/
kaggle datasets download -d johnsmith88/heart-disease-dataset -p ./data/
```

### Option B: Manual Download
1. Go to the Kaggle links above
2. Click "Download" 
3. Extract to this `data/` directory

## Note
These datasets are NOT included in the repository due to size.
The app will work in **demo mode** without them (using pretrained/randomly generated models).
