# Task 4: Loan Approval Prediction

## Overview
This project develops a classification model to predict loan approval based on applicant details such as age, income, credit score, and more. The notebook focuses on handling categorical and numerical features, training multiple classifiers (Logistic Regression, Decision Tree, Random Forest), evaluating performance with metrics like precision, recall, and F1-score, and analyzing feature importances. Although the objectives mention SMOTE for imbalance and missing value handling, the code primarily uses preprocessing pipelines and drops duplicates; imbalance handling is not implemented but can be added.

## Objectives
- Predict loan approval (using "Previous_Defaults" as proxy target) based on applicant information.
- Properly encode categorical variables and scale numerical features.
- Evaluate models with precision, recall, F1-score, and confusion matrices.
- Identify key features via importance analysis.
- (Bonus) Perform hyperparameter tuning on Random Forest for optimization.

## Dataset
- **File**: `loan_approval_dataset.csv`
- **Source**: Not specified (assumed provided or synthetic/publicly available).
- **Key Columns**:
  - Numerical: `Age`, `Income`, `Credit_Score`, `Loan_Amount`, `Loan_Term`, `Interest_Rate`, `Debt_to_Income_Ratio`, `Number_of_Dependents`.
  - Categorical: `Employment_Status`, `Marital_Status`, `Property_Ownership`, `Loan_Purpose`.
  - Target: `Previous_Defaults` (binary or multi-class; used as approval proxy).
  - Other: `Applicant_ID` (dropped implicitly).
- **Size**: 5,000 entries, 14 columns.
- **Preprocessing**: Drop duplicates, one-hot encoding for categorical features, standard scaling for numerical features. No explicit missing value handling in code.

## Requirements
- Python 3.12+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn (sklearn)

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Ensure the dataset (`loan_approval_dataset.csv`) is in the same directory as the notebook.

## Usage
1. Open the Jupyter notebook: `Task 4 - Loan Approval Prediction Description.ipynb`.
2. Run the cells sequentially:
   - Data loading and cleaning.
   - Feature selection and train/test split.
   - Preprocessing pipeline (encoding and scaling).
   - Train and evaluate models (Logistic Regression, Decision Tree, Random Forest).
   - Visualize confusion matrices and feature importances.
   - (Bonus) Hyperparameter tuning with GridSearchCV on Random Forest.
3. Customize:
   - Add SMOTE to the pipeline for imbalance handling (e.g., via `imblearn`).
   - Adjust hyperparameters in GridSearchCV.
   - Experiment with additional models like XGBoost.

Example command to start Jupyter:
```
jupyter notebook
```

## Results
- **Logistic Regression**:
  - Accuracy: ~90%
  - F1-Score (macro avg): ~0.47 (imbalanced performance on minority class).
- **Decision Tree**:
  - Accuracy: ~90%
  - F1-Score (macro avg): ~0.47
- **Random Forest**:
  - Accuracy: ~90%
  - F1-Score (macro avg): ~0.47
- **Tuned Random Forest** (Bonus):
  - Best Params: {'clf__max_depth': 5, 'clf__min_samples_split': 2, 'clf__n_estimators': 100}
  - CV Accuracy: 0.9005
- **Key Insights**: Models perform well overall but struggle with minority class recall due to potential imbalance. Top features include Credit_Score, Debt_to_Income_Ratio, and Income.

## Visualization
- **Confusion Matrices**: Heatmaps for each model showing true positives/negatives.
- **Feature Importances**: Bar plots of top 15 features for Decision Tree and Random Forest.

## Conclusion
The Random Forest model, after tuning, provides reliable predictions for loan approvals with ~90% accuracy. Preprocessing ensures robust handling of mixed data types. To improve minority class performance, incorporate SMOTE as per objectives. Key predictors like credit score highlight areas for business focus. Extend with advanced techniques like ensemble voting or feature engineering for better results.

## License
This project is for educational purposes. No specific license is applied.

## Author
Nazia Shar

## Contact
For questions or contributions, feel free to open an issue or pull request on GitHub.