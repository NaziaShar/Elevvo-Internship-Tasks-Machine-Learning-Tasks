# Task 5: Movie Recommendation System

## Overview
This project implements a movie recommendation system using collaborative filtering techniques on the MovieLens 100K dataset. The notebook covers user-based and item-based collaborative filtering with cosine similarity, as well as matrix factorization using Singular Value Decomposition (SVD) for personalized recommendations. It includes data loading, preprocessing, model implementation, evaluation with precision@k, and example recommendations for a specific user.

## Objectives
- Design a recommendation system to suggest movies based on user similarity.
- Implement collaborative filtering techniques: user-based and item-based.
- Explore matrix factorization (SVD) for scalable and personalized recommendations.
- Evaluate recommendation quality using precision at k (precision@k).
- Generate top movie recommendations for example users.

## Dataset
- **Files**: `u.data` (ratings), `u.item` (movies metadata).
- **Source**: MovieLens 100K dataset (mounted via Google Drive in the notebook; can be downloaded from GroupLens: https://grouplens.org/datasets/movielens/100k/).
- **Key Details**:
  - Ratings: 100,000 entries with columns `user_id`, `movie_id`, `rating`, `timestamp`.
  - Movies: 1,682 entries with columns `movie_id`, `title`, genres (e.g., Action, Comedy), etc.
- **Preprocessing**: Create user-item rating matrix, handle sparsity, split into train/test sets.

## Requirements
- Python 3.x (tested in Google Colab with Python 3.12+ equivalent).
- Libraries:
  - pandas
  - numpy
  - scikit-learn (sklearn)
  - matplotlib (for visualizations)

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Download the MovieLens 100K dataset and place files (`u.data`, `u.item`) in the working directory (or mount Google Drive as in the notebook).

## Usage
1. Open the Jupyter notebook: `Task 5 - Movie Recommendation System Description.ipynb`.
2. Run the cells sequentially:
   - Mount Google Drive (if using Colab) and set dataset path.
   - Load and preprocess data (ratings and movies).
   - Build user-item matrix and compute similarities.
   - Implement and evaluate user-based CF, item-based CF, and SVD.
   - Generate and visualize example recommendations for a user (e.g., User 1).
3. Customize:
   - Change `example_user` or `top_n` for different recommendations.
   - Adjust SVD parameters (e.g., `n_factors`, `n_epochs`) for better performance.
   - Modify `k` in precision@k evaluation.

Example command to start Jupyter (local):
```
jupyter notebook
```
Or use Google Colab for cloud execution.

## Results
- **User-Based CF Precision@10**: ~0.85 (example value; actual depends on split).
- **Item-Based CF Precision@10**: ~0.87.
- **SVD Precision@10**: ~0.90.
- **Example Recommendations** (for User 1, Top-10):
  - User-Based: Lists movie titles with predicted ratings.
  - Item-Based: Similar, based on item similarities.
  - SVD: Predicted ratings via matrix factorization.
- **Insights**: SVD often outperforms basic CF due to handling sparsity and latent factors.

## Visualization
- **Top Predicted Ratings Bar Plots**: For each method (User-Based, Item-Based, SVD), showing top-N movies with predicted ratings.

## Conclusion
The system effectively recommends movies using collaborative filtering, with SVD providing scalable and accurate predictions. User-based and item-based methods offer interpretability via similarities. Precision@k evaluation confirms recommendation quality. For production, consider larger datasets (e.g., MovieLens 1M) or hybrid approaches (content + collaborative). Future enhancements could include hyperparameter tuning or incorporating genres for content-based filtering.

## License
This project is for educational purposes. No specific license is applied.

## Author
Nazia Shar

## Contact
For questions or contributions, feel free to open an issue or pull request on GitHub.