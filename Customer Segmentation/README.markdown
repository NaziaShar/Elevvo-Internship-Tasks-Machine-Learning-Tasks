# Task 2: Customer Segmentation

## Overview
This project focuses on segmenting customers based on demographic and behavioral data using unsupervised machine learning techniques. The primary goal is to identify distinct customer groups to enable targeted marketing strategies. The notebook implements K-Means clustering, determines the optimal number of clusters using the Elbow method and Silhouette score, and provides insights into customer segments. Although DBSCAN is mentioned in the objectives, the current implementation emphasizes K-Means for segmentation.

## Objectives
- Group customers into meaningful clusters using features like age, annual income, and spending score.
- Determine the optimal number of clusters via the Elbow method and Silhouette score.
- Explore clustering algorithms (primarily K-Means, with potential for DBSCAN comparison).
- Analyze cluster characteristics and predict segments for new customers.
- Generate actionable insights for marketing based on segments.

## Dataset
- **File**: `Mall_Customers.csv`
- **Source**: Assumed to be provided or publicly available (e.g., from Kaggle or similar repositories).
- **Key Columns**:
  - Numerical Features: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.
  - Other: `CustomerID`, `Gender` (not used in clustering but available for extension).
- **Size**: 200 entries (based on typical Mall_Customers dataset; confirm with actual file).
- **Preprocessing**: Data cleaning (dropping duplicates, stripping column names), feature selection, and standardization using `StandardScaler`.

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
3. Ensure the dataset (`Mall_Customers.csv`) is in the same directory as the notebook.

## Usage
1. Open the Jupyter notebook: `Task 2 - Customer Segmentation.ipynb`.
2. Run the cells sequentially:
   - Data loading and cleaning.
   - Preprocessing (scaling).
   - Optimal cluster determination (Elbow and Silhouette).
   - Model training with K-Means (k=5).
   - Visualization of clusters.
   - Cluster analysis (mean values).
   - Simulated customer prediction and insights.
3. Customize:
   - Adjust the `K_range` for Elbow method exploration.
   - Modify `final_k` based on analysis.
   - Add DBSCAN implementation in a new cell for comparison (e.g., using `sklearn.cluster.DBSCAN`).
   - Test different customer profiles in the prediction section.

Example command to start Jupyter:
```
jupyter notebook
```

## Results
- **Optimal Clusters**: Determined as 5 using Elbow method (inertia plot) and Silhouette scores.
- **Cluster Means** (example output):
  ```
               Age  Annual Income (k$)  Spending Score (1-100)
  Cluster                                                      
  0        55.28           47.62               41.71
  1        32.88           86.10               81.53
  2        25.77           26.12               74.85
  3        26.73           54.31               40.91
  4        44.39           89.77               18.48
  ```
- **Example Prediction**:
  - Input: Age=30, Income=60k$, Spending=80
  - Predicted Cluster: 1
  - Characteristics: Average Age ~32.88, Income ~86.1k$, Spending ~81.53, Size=40
  - Insights: Premium Customers – High income, High spending. Best for luxury products.
- **Evaluation**: Silhouette score used for cluster quality; visualizations confirm separation.

## Visualization
- **Cluster Scatterplot**: Displays customer segments based on Age vs. Annual Income, colored by cluster.
- Additional plots (add in notebook if needed): Pairplots for multi-feature views or 3D scatter for all features.

## Conclusion
The K-Means model effectively segments customers into 5 groups, revealing patterns like premium high-spenders and budget-conscious groups. This enables personalized marketing, such as targeting high-income clusters with luxury offers. Future enhancements could include DBSCAN for density-based clustering, incorporating more features (e.g., Gender), or using advanced techniques like hierarchical clustering.

## License
This project is for educational purposes. No specific license is applied.

## Author
Nazia Shar

## Contact
For questions or contributions, feel free to open an issue or pull request on GitHub.