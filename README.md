# 🧬 Forensic DNA Profile Matching Using 1D Convolutional Neural Networks (CNN)

> **An end-to-end deep learning pipeline for forensic DNA profile matching using Short Tandem Repeat (STR) markers.**

---

## 📌 Project Overview

DNA profile matching is a fundamental task in forensic science, helping investigators determine whether two DNA samples originate from the same individual.

This project develops a **1D Convolutional Neural Network (CNN)** to classify DNA profile pairs as **Match** or **No Match** using STR (Short Tandem Repeat) markers. The model is trained on preprocessed forensic DNA data and evaluated against traditional machine learning models, demonstrating improved predictive performance.

A **Streamlit web application** is also included for real-time DNA match prediction.

---

## 🎯 Objectives

- Build an automated DNA profile matching system.
- Classify DNA sample pairs as **Match** or **No Match**.
- Compare CNN performance with traditional ML algorithms.
- Evaluate model performance using multiple classification metrics.
- Deploy the trained model through a Streamlit application.

---

## 📊 Dataset

**Domain:** Forensic DNA Analysis

### Data Processing

- Data Collection
- Missing Value Handling
- Feature Encoding
- Normalization
- Train-Test Split

The dataset consists of DNA profiles represented by **Short Tandem Repeat (STR) markers**, commonly used in forensic identification.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |

---

## 🧠 Model Architecture

The project uses a custom **1D Convolutional Neural Network** consisting of:

- Input Layer
- 1D Convolution Layers
- Max Pooling Layers
- Dropout Layers
- Fully Connected Dense Layers
- Sigmoid Output Layer

---

## 🔄 Project Workflow

```
DNA Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Encoding
      │
      ▼
Normalization
      │
      ▼
Train/Test Split
      │
      ▼
1D CNN Model
      │
      ▼
Model Evaluation
      │
      ▼
Performance Comparison
      │
      ▼
Streamlit Deployment
```

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Test Accuracy | **86%** |
| AUC-ROC | **0.9433** |
| Classification | Binary (Match / No Match) |

### Baseline Comparison

The CNN model outperformed traditional machine learning models including:

- Support Vector Machine (SVM)
- Random Forest

with an improvement of approximately **8 percentage points** in overall accuracy.

---

## 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC Score
- Confusion Matrix

---

## 💻 Streamlit Application

The project includes an interactive Streamlit application that allows users to:

- Upload DNA profile data
- Predict Match / No Match
- View prediction confidence
- Perform real-time inference

---

## 📂 Project Structure

```
Forensic-DNA-CNN/
│
├── dataset/
│   └── dna_profiles.csv
│
├── notebooks/
│   └── forensic_dna_cnn.ipynb
│
├── models/
│   └── cnn_model.keras
│
├── app/
│   └── streamlit_app.py
│
├── images/
│   ├── cnn_architecture.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_accuracy.png
│   └── streamlit_demo.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/madhavamanimaran/forensic-dna-cnn.git
```

### Navigate to Project

```bash
cd forensic-dna-cnn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook
```

### Launch Streamlit App

```bash
streamlit run app.py
```

---

## 📷 Results

Include screenshots of:

- Model Architecture
- ROC Curve
- Confusion Matrix
- Training Curves
- Streamlit Interface

Example:

```markdown
![ROC Curve](images/roc_curve.png)

![Confusion Matrix](images/confusion_matrix.png)

![Streamlit App](images/streamlit_demo.png)
```

---

## 🔮 Future Enhancements

- Multi-class DNA profile classification
- Explainable AI (Grad-CAM / SHAP)
- Hyperparameter optimization
- Cloud deployment
- Larger forensic DNA datasets

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Madhavamanimaran**

**Python | AI | Machine Learning | Deep Learning**

- 💼 LinkedIn: https://www.linkedin.com/in/madhavamanimaran


---

⭐ If you found this project interesting, consider giving it a **Star** on GitHub!
