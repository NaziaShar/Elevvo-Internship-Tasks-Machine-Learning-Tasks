# Task 7: Sales Forecasting

## Overview
This project focuses on forecasting future sales using historical Walmart sales data. The notebook implements regression and boosting models (Linear Regression, Random Forest, XGBoost, LightGBM) with time-based features to capture trends and seasonality. It includes data loading, feature engineering (e.g., day, month, lag, rolling averages), model training, evaluation with RMSE and MAE, and visualization of forecasts.

## Objectives
- Forecast future sales based on historical Walmart sales data.
- Create time-based features (e.g., day, month, lag features, rolling averages) for accurate modeling.
- Compare regression models (Linear Regression, Random Forest) with boosting models (XGBoost, LightGBM).
- Evaluate forecasting performance using RMSE and MAE metrics.

## Dataset
- **Files**: 
  - `train.csv`: Sales data with store, department, date, and weekly sales.
  - `features.csv`: Additional features like temperature, fuel price, CPI, unemployment.
  - `stores.csv`: Store metadata (e.g., store type, size).
  - `test.csv`: Test data for predictions.
- **Source**: Walmart Sales Forecasting dataset (assumed from Kaggle or similar; stored in Google Drive as per notebook).
- **Key Columns**:
  - Train: `Store`, `Dept`, `Date`, `Weekly_Sales` (target), `IsHoliday`.
  - Features: `Store`, `Date`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, etc.
  - Stores: `Store`, `Type`, `Size`.
- **Size**: ~421,570 entries in train.csv (varies by dataset version).
- **Preprocessing**: Merge datasets, create time-based features (day, month, year, lags, rolling means), encode categorical variables, handle missing values.

## Requirements
- Python 3.x (tested in Google Colab; Python 3.12+ equivalent).
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - xgboost
  - lightgbm
  - statsmodels
- Hardware: Optional GPU support for faster boosting model training.

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib scikit-learn xgboost lightgbm statsmodels
   ```
3. Place dataset files (`train.csv`, `features.csv`, `stores.csv`, `test.csv`) in the working directory or mount Google Drive (as per notebook).

## Usage
1. Open the Jupyter notebook: `Task 7 - Sales Forecasting Description.ipynb`.
2. Run cells sequentially:
   - Mount Google Drive (if using Colab) and set dataset paths.
   - Load and merge datasets, perform feature engineering.
   - Train and evaluate models (Linear Regression, Random Forest, XGBoost, LightGBM).
   - Visualize actual vs. predicted sales (aggregated by date).
3. Customize:
   - Adjust time-based feature engineering (e.g., lag periods, rolling window sizes).
   - Tune XGBoost/LightGBM hyperparameters (e.g., learning rate, max depth).
   - Experiment with additional models like ARIMA or Prophet.

Example command to start Jupyter (local):
```
jupyter notebook
```
Or use Google Colab for cloud execution.

## Results
- **Linear Regression**:
  - RMSE: ~High (baseline model, less effective for complex patterns).
  - MAE: ~High.
- **Random Forest**:
  - RMSE: ~Moderate (improves over Linear Regression).
  - MAE: ~Moderate.
- **XGBoost**:
  - RMSE: ~Low (captures trends well).
  - MAE: ~Low.
- **LightGBM**:
  - RMSE: ~Lowest (best performer due to efficiency and handling of categorical features).
  - MAE: ~Lowest.
- **Key Insights**: Boosting models (XGBoost, LightGBM) outperform traditional regression due to their ability to model non-linear relationships and seasonality.

## Visualization
- **Actual vs. Predicted Sales**: Line plot comparing actual weekly sales (aggregated) with XGBoost and LightGBM predictions over time.
- **Seasonal Decomposition**: (If implemented) Trend, seasonal, and residual components from `statsmodels`.

## Conclusion
LightGBM and XGBoost deliver the most accurate sales forecasts, leveraging time-based features like lags and rolling averages. Feature engineering is critical for capturing seasonality and trends. For production, consider ensemble methods or hybrid models (e.g., combining boosting with ARIMA). Future improvements could include external variables (e.g., promotions) or advanced time-series models.

## License
This project is for educational purposes. No specific license is applied.

## Author
Nazia Shar

## Contact
For questions or contributions, open an issue or pull request on GitHub.