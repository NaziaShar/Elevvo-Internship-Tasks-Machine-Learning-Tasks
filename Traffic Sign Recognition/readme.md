# GTSRB Traffic Sign Recognition with DenseNet121

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-High-success)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📌 Project Overview
This project implements a **Traffic Sign Recognition System** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The model is based on a **pretrained DenseNet121** architecture, fine-tuned for multi-class image classification.

Traffic sign recognition is a key component in driver assistance systems and autonomous vehicles, enabling real-time road safety compliance.

---

## 📂 Dataset
- **Dataset**: [GTSRB - German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html)
- **Classes**: 43 traffic sign categories
- **Image size**: Resized to **32x32 pixels**
- **Split**:
  - Training data: `Train/`
  - Test data: `Test/`

---

## 🛠 Dependencies
Make sure the following libraries are installed:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn
```

---

## ⚙️ Project Workflow

### 1. Data Loading & Preprocessing
- Load images from dataset folders.
- Resize all images to **32x32**.
- Normalize pixel values.
- Convert labels to categorical (one-hot encoding).

### 2. Model Architecture
- Pretrained **DenseNet121** as feature extractor.
- Added custom classification layers:
  - Global Average Pooling
  - Dense (Fully Connected)
  - Softmax for 43 classes

### 3. Training
- Optimizer: **Adam**
- Loss: **Categorical Crossentropy**
- Callbacks:
  - EarlyStopping (to prevent overfitting)
  - ReduceLROnPlateau (learning rate scheduling)

### 4. Evaluation
- Metrics: Accuracy, Confusion Matrix
- Visualization:
  - Training & validation accuracy/loss curves
  - Confusion matrix heatmap

---

## 🚀 Usage

### 1. Clone Repository
```bash
git clone https://github.com/your-username/gtsrb-densenet121.git
cd gtsrb-densenet121
```

### 2. Run Notebook
Open Jupyter Notebook / Colab and run:
```bash
jupyter notebook gtsrb-pretrained-densenet121-model-cnn.ipynb
```

---

## 📊 Results

- **Custom CNN (with Augmentation)**  
  - Training Accuracy: **99%**  
  - Validation Accuracy: **93%**  

- **DenseNet121 (without Augmentation)**  
  - Training Accuracy: **100%**  
  - Validation Accuracy: **95%**  

- **DenseNet121 (with Augmentation)**  
  - Training Accuracy: **99%**  
  - Validation Accuracy: **96%**  

---

## 📌 Future Improvements
- Experiment with other architectures (ResNet, MobileNetV2, EfficientNet).
- Deploy as a **Flask/Streamlit web app**.
- Optimize for **real-time inference** on embedded devices.

---

## 👨‍💻 Author
Developed by **[Your Name]**  
📧 Contact: your_email@example.com

