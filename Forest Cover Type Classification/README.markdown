# Task 3: Forest Cover Type Classification

## Overview
This project focuses on classifying forest cover types into seven distinct categories using cartographic and environmental features. The implementation leverages tree-based models, primarily Random Forest and Decision Tree, for multi-class classification, with Logistic Regression as a baseline. The notebook includes data loading, cleaning, preprocessing with SMOTE for imbalance handling, model training, evaluation, and feature importance analysis to identify key predictors.

## Objectives
- Classify forest cover types (7 classes) using features like elevation, slope, and soil types.
- Implement and compare tree-based models (Decision Tree, Random Forest) for classification.
- Evaluate model performance using precision, recall, F1-score, and accuracy.
- Identify the most influential features through feature importance analysis.
- Address class imbalance using SMOTE oversampling.

## Dataset
- **File**: `covtype.csv`
- **Source**: UCI Machine Learning Repository or equivalent (e.g., Kaggle).
- **Key Columns**:
  - Numerical: `Elevation`, `Aspect`, `Slope`, `Horizontal_Distance_To_Hydrology`, `Vertical_Distance_To_Hydrology`, `Horizontal_Distance_To_Roadways`, `Hillshade_9am`, `Hillshade_Noon`, `Hillshade_3pm`, `Horizontal_Distance_To_Fire_Points`.
  - Binary: 4 wilderness areas, 40 soil types (one-hot encoded).
  - Target: `Cover_Type` (1-7, representing forest types like Spruce/Fir, Lodgepole Pine).
- **Size**: 581,012 entries, 55 columns.
- **Preprocessing**: Drop duplicates, standardize numerical features, apply SMOTE for class imbalance.

## Requirements
- Python 3.12+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - seaborn
  - matplotlib

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
   ```
3. Place `covtype.csv` in the same directory as the notebook. (Download from: https://archive.ics.uci.edu/dataset/31/covertype)

## Usage
1. Open the Jupyter notebook: `Task 3 - Forest Cover Type Classification.ipynb`.
2. Run cells sequentially:
   - Load and clean data.
   - Split data and preprocess (scaling, SMOTE).
   - Train and evaluate Random Forest, Decision Tree, and Logistic Regression models.
   - Visualize confusion matrix and feature importances.
3. Customize:
   - Modify `n_estimators` in Random Forest or `max_depth` in Decision Tree.
   - Experiment with SMOTE parameters or other resampling methods.
   - Add hyperparameter tuning with GridSearchCV.

Example command to start Jupyter:
```
jupyter notebook
```

## Results
- **Random Forest**:
  - Accuracy: 96%
  - Macro Avg F1: 0.93
  - Key Features: `Elevation`, `Horizontal_Distance_To_Fire_Points`, `Horizontal_Distance_To_Roadways`.
- **Decision Tree**:
  - Accuracy: 94%
  - Macro Avg F1: 0.90
- **Logistic Regression** (Baseline):
  - Accuracy: 60%
  - Macro Avg F1: 0.51
- **Visualizations**:
  - Confusion matrix heatmap for Random Forest.
  - Bar plot of top 15 feature importances.

## Conclusion
Random Forest achieves the highest performance (96% accuracy) due to its robustness with high-dimensional, imbalanced data. SMOTE effectively balances minority classes, improving recall for rare cover types. Elevation is the most predictive feature. Future work could explore gradient boosting (e.g., XGBoost) or neural networks for further gains.

## License
This project is for educational purposes. No specific license is applied.

## Author
Nazia Shar

## Contact
For questions or contributions, open an issue or pull request on GitHub.