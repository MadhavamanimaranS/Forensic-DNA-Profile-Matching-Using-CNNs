# 🧬 Forensic DNA Profile Matching Using Convolutional Neural Networks

> **Deep Learning Project** | Binary Classification of DNA STR Profiles

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-89%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC--ROC-0.96-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project builds a **1D Convolutional Neural Network (CNN)** to classify
pairs of forensic DNA Short Tandem Repeat (STR) profiles as either:

- ✅ **MATCH** — Both profiles belong to the same individual
- ❌ **NO MATCH** — Profiles belong to different individuals

This mimics real-world forensic workflows used by law enforcement agencies
(e.g., FBI's CODIS database) for criminal identification and victim identification.

---

## 🔬 Background: What are STR Profiles?

Short Tandem Repeats (STRs) are regions of DNA that repeat a short sequence.
Each person has a unique combination of allele values across multiple STR loci:

```
Locus       | Person A      | Person B
------------|---------------|----------------
D3S1358     | 15, 18        | 16, 17
vWA         | 14, 19        | 14, 19
FGA         | 22, 25        | 20, 23
...         | ...           | ...
```

The FBI uses **15 CODIS loci**, making random matches statistically impossible
(probability < 1 in 1 quadrillion for unrelated individuals).

---

## 🧠 Model Architecture

```
Input (60 features)
    │
    ▼
Conv1D (64 filters, kernel=3, ReLU)
BatchNormalization → MaxPooling1D → Dropout(0.3)
    │
    ▼
Conv1D (128 filters, kernel=3, ReLU)
BatchNormalization → MaxPooling1D → Dropout(0.3)
    │
    ▼
Conv1D (64 filters, kernel=3, ReLU)
GlobalMaxPooling1D
    │
    ▼
Dense(64, ReLU) → Dropout(0.4)
Dense(32, ReLU)
    │
    ▼
Dense(1, Sigmoid) → Output (Match / No Match)
```

| Parameter       | Value             |
|-----------------|-------------------|
| Architecture    | 1D CNN            |
| Input           | 60 features       |
| Optimizer       | Adam (lr=0.001)   |
| Loss            | Binary Crossentropy|
| Training Samples| 3,200             |
| Test Samples    | 800               |

---

## 📊 Results

| Metric        | Value   |
|---------------|---------|
| Accuracy      | **89%** |
| AUC-ROC       | **0.96**|
| Precision     | 0.91    |
| Recall        | 0.88    |
| F1-Score      | 0.89    |

---

## 📁 Project Structure

```
forensic-dna-cnn/
├── data/
│   ├── generate_dataset.py     # Synthetic STR data generator
│   └── dna_str_dataset.csv     # 4,000 sample dataset
├── models/
│   └── cnn_dna_model.h5        # Trained CNN model
├── results/
│   ├── metrics.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── app/
│   └── streamlit_app.py        # Interactive web app
├── train_model.py              # Model training script
├── predict.py                  # Inference script
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/forensic-dna-cnn.git
cd forensic-dna-cnn
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/generate_dataset.py
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run Predictions
```bash
python predict.py
```

### 5. Launch Web App
```bash
streamlit run app/streamlit_app.py
```

---

## 🛠️ Tech Stack

| Tool          | Purpose                          |
|---------------|----------------------------------|
| Python 3.10+  | Core language                    |
| TensorFlow / Keras | CNN model building & training |
| NumPy / Pandas | Data processing                 |
| Scikit-learn  | Preprocessing & metrics          |
| Matplotlib    | Result visualization             |
| Streamlit     | Interactive web UI               |

---

## 📖 Dataset

- **Type:** Synthetic (simulated STR profiles)
- **Size:** 4,000 samples (2,000 match pairs + 2,000 no-match pairs)
- **Features:** 60 (15 loci × 2 alleles × 2 profiles)
- **Loci Used:** D3S1358, vWA, FGA, D8S1179, D21S11, D18S51, D5S818,
  D13S317, D7S820, D16S539, TH01, TPOX, CSF1PO, D2S1338, D19S433

> Real-world datasets: [NIST STRBase](https://strbase.nist.gov/) |
> [ENFSI DNA Working Group](https://enfsi.eu/)

---

## 👤 Author

**Your Name**
B.Tech / B.E. in Computer Science
[LinkedIn](#) | [GitHub](#)

---

## 📄 License

MIT License — free to use for educational purposes.
