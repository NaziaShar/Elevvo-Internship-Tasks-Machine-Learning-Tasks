# Task 1: Student Score Prediction

## Overview
This project involves building a regression model to predict students' exam scores based on various factors, with a primary focus on study hours. The analysis explores the relationship between study time and academic performance using simple linear regression and compares it to polynomial regression for potential accuracy improvements. The notebook includes data loading, exploratory data analysis (EDA), model training, evaluation, and feature experimentation.

## Objectives
- Develop a regression model to predict exam scores using study hours and other student-related features.
- Analyze the correlation between study time and exam performance.
- Compare the performance of simple linear regression with polynomial regression (degree 2).
- Experiment with different feature combinations (e.g., removing sleep-related features) to assess impact on model accuracy.

## Dataset
- **File**: `StudentPerformanceFactors.csv`
- **Source**: Not specified (assumed to be provided or publicly available).
- **Key Columns**:
  - Numerical: `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`, `Exam_Score` (target).
  - Categorical: `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`.
- **Size**: 6607 entries, 20 columns.
- **Preprocessing**: Handles missing values (imputation), scaling for numerical features, and one-hot encoding for categorical features.

## Requirements
- Python 3.12+
- Libraries:
  - pandas
  - numpy
  - scikit-learn (sklearn)
  - matplotlib

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Ensure the dataset (`StudentPerformanceFactors.csv`) is in the same directory as the notebook.

## Usage
1. Open the Jupyter notebook: `Task 1- Student Score Prediction (2).ipynb`.
2. Run the cells sequentially:
   - Data loading and EDA.
   - Model training (linear and polynomial regression).
   - Evaluation and visualizations.
   - Feature experimentation (e.g., removing sleep-related features).
3. Customize:
   - Adjust the `test_size` or `random_state` in `train_test_split`.
   - Experiment with different `features_to_remove` for feature ablation studies.
   - Modify polynomial degree in the bonus section.

Example command to start Jupyter:
```
jupyter notebook
```

## Results
- **Linear Regression**:
  - MAE: ~0.452
  - RMSE: ~1.804
  - R²: ~0.77
- **Polynomial Regression (Degree 2)**:
  - MAE: ~0.656
  - RMSE: ~1.895
  - R²: ~0.746
- **Feature Experiment (Removing Sleep Features)**:
  - Slight increase in MAE (~0.002), indicating sleep hours may have a minor positive impact on predictions.
- Visualizations:
  - Correlation heatmap.
  - Actual vs. Predicted scatter plot.
  - Residuals histogram.

## Conclusion
The linear regression model performs well in predicting exam scores, with study hours and attendance showing strong correlations. Polynomial regression (degree 2) does not significantly improve accuracy, suggesting a linear relationship dominates. Removing less influential features (e.g., sleep hours) has minimal impact, but further hyperparameter tuning or advanced models (e.g., random forests) could enhance results.

## License
This project is for educational purposes. No specific license is applied.

## Author
Nazia Shar

## Contact
For questions or contributions, feel free to open an issue or pull request on GitHub.