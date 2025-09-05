# Elevvo Internship Machine Learning Projects
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-High-success)

## Overview
This repository contains a collection of machine learning projects demonstrating various techniques, including regression, classification, clustering, recommendation systems, and deep learning. Each project (Task 1 through Task 7 and an additional Traffic Sign Classification project) addresses a unique problem, leveraging datasets from sources like Kaggle, UCI, or GroupLens. The projects showcase skills in data preprocessing, feature engineering, model training, evaluation, and visualization, with some utilizing GPU acceleration on Kaggle for compute-intensive tasks.

## Projects
Below is a summary of the individual tasks included in this repository, each located in its own subdirectory with a detailed README.md file.

1. **Task 1: Student Score Prediction**
   - **Objective**: Predict student exam scores using features like study hours and attendance.
   - **Techniques**: Linear and polynomial regression, feature ablation.
   - **Dataset**: `StudentPerformanceFactors.csv` (6,607 entries).
   - **Key Results**: Linear regression achieved ~77% R²; study hours strongly correlate with scores.
   - **Directory**: `/Task1_Student_Score_Prediction`

2. **Task 2: Customer Segmentation**
   - **Objective**: Segment customers based on age, income, and spending score for targeted marketing.
   - **Techniques**: K-Means clustering, Elbow method, Silhouette score.
   - **Dataset**: `Mall_Customers.csv` (~200 entries).
   - **Key Results**: Identified 5 clusters, with premium customers (high income, high spending) suitable for luxury products.
   - **Directory**: `/Task2_Customer_Segmentation`

3. **Task 3: Forest Cover Type Classification**
   - **Objective**: Classify forest cover types (7 classes) using environmental features.
   - **Techniques**: Random Forest, Decision Tree, Logistic Regression, SMOTE for imbalance.
   - **Dataset**: `covtype.csv` (581,012 entries, UCI).
   - **Key Results**: Random Forest achieved 96% accuracy; elevation was the top predictor.
   - **Directory**: `/Task3_Forest_Cover_Type`

4. **Task 4: Loan Approval Prediction**
   - **Objective**: Predict loan approval based on applicant details like credit score and income.
   - **Techniques**: Logistic Regression, Decision Tree, Random Forest, hyperparameter tuning.
   - **Dataset**: `loan_approval_dataset.csv` (5,000 entries).
   - **Key Results**: Random Forest achieved ~90% accuracy; credit score was a key feature.
   - **Directory**: `/Task4_Loan_Approval_Prediction`

5. **Task 5: Movie Recommendation System**
   - **Objective**: Build a recommendation system for movies based on user preferences.
   - **Techniques**: User-based and item-based collaborative filtering, SVD matrix factorization.
   - **Dataset**: MovieLens 100K (`u.data`, `u.item`, 100,000 ratings).
   - **Key Results**: SVD achieved ~90% precision@10, outperforming basic collaborative filtering.
   - **Directory**: `/Task5_Movie_Recommendation`

6. **Task 6: Music Genre Classification**
   - **Objective**: Classify music tracks into genres using audio features like MFCCs.
   - **Techniques**: Random Forest, CNNs, spectrogram-based deep learning.
   - **Dataset**: `features_30_sec.csv` (GTZAN, ~1,000 entries).
   - **Key Results**: CNNs achieved ~75-85% accuracy, outperforming Random Forest (~65-75%).
   - **Directory**: `/Task6_Music_Genre_Classification`

7. **Task 7: Sales Forecasting**
   - **Objective**: Forecast Walmart weekly sales using historical data and time-based features.
   - **Techniques**: Linear Regression, Random Forest, XGBoost, LightGBM.
   - **Dataset**: Walmart Sales (`train.csv`, `features.csv`, `stores.csv`, ~421,570 entries).
   - **Key Results**: LightGBM achieved the lowest RMSE; time-based features were critical.
   - **Directory**: `/Task7_Sales_Forecasting`

8. **Task 8: Traffic Sign Classification with Pretrained CNN**
   - **Objective**: Classify German traffic signs (43 classes) using image data.
   - **Techniques**: Pretrained CNN (e.g., VGG16, ResNet) with data augmentation, GPU acceleration on Kaggle.
   - **Dataset**: GTSRB (~39,209 training, ~12,630 test images, Kaggle).
   - **Key Results**: Baseline accuracy of 55.99%; GPU reduced training time significantly.
   - **Directory**: `/Traffic_Sign_Classification`

## Requirements
- **Python**: 3.11+ (Task 8) or 3.12+ (Tasks 1-7).
- **Libraries**:
  - Core: pandas, numpy, matplotlib, seaborn, scikit-learn
  - Deep Learning: tensorflow (Tasks 6, 8)
  - Boosting: xgboost, lightgbm (Task 7)
  - Others: imbalanced-learn (Task 3), statsmodels (Task 7), joblib (Task 6)
- **Hardware**: GPU recommended for Tasks 6 and 8 (e.g., Kaggle P100/T4 or local CUDA-compatible GPU).

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Install dependencies (use a virtual environment for isolation):
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost lightgbm imbalanced-learn statsmodels joblib
   ```
3. Download datasets and place them in the respective task directories:
   - Task 1: `StudentPerformanceFactors.csv`
   - Task 2: `Mall_Customers.csv`
   - Task 3: `covtype.csv` (https://archive.ics.uci.edu/dataset/31/covertype)
   - Task 4: `loan_approval_dataset.csv`
   - Task 5: MovieLens 100K (`u.data`, `u.item`, https://grouplens.org/datasets/movielens/100k/)
   - Task 6: `features_30_sec.csv` (GTZAN)
   - Task 7: Walmart Sales (`train.csv`, `features.csv`, `stores.csv`, `test.csv`)
   - Task 8: GTSRB (https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
4. For GPU support (Tasks 6, 8), use Kaggle or ensure a local CUDA-compatible GPU with TensorFlow GPU setup.

## Usage
1. Navigate to the desired task directory (e.g., `/Task1_Student_Score_Prediction`).
2. Open the Jupyter notebook (e.g., `Task 1- Student Score Prediction (2).ipynb`) in Jupyter or Kaggle (for Task 8).
3. Run cells sequentially to:
   - Load and preprocess data.
   - Train and evaluate models.
   - Visualize results (e.g., plots, confusion matrices).
4. For Task 8, use Kaggle’s GPU environment for faster training:
   - Upload the notebook to Kaggle, enable GPU, and execute.
5. Refer to individual task READMEs for specific customization options (e.g., hyperparameter tuning, feature engineering).

Example command to start Jupyter locally:
```
jupyter notebook
```

## Project Structure
```
your-repo-name/
├── Task1_Student_Score_Prediction/
│   ├── Task 1- Student Score Prediction (2).ipynb
│   ├── README.md
├── Task2_Customer_Segmentation/
│   ├── Task 2 - Customer Segmentation.ipynb
│   ├── README.md
├── Task3_Forest_Cover_Type/
│   ├── Task 3 - Forest Cover Type Classification.ipynb
│   ├── README.md
├── Task4_Loan_Approval_Prediction/
│   ├── Task 4 - Loan Approval Prediction Description.ipynb
│   ├── README
