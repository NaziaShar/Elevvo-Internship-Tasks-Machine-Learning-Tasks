# Task 6: Music Genre Classification

## Overview
This project focuses on classifying music tracks into genres using audio features such as Mel-Frequency Cepstral Coefficients (MFCCs). The notebook compares traditional machine learning models (Random Forest) with deep learning approaches (Convolutional Neural Networks, CNNs) and evaluates the potential of transfer learning and spectrogram-based methods. It includes data loading, preprocessing, model training, evaluation with classification metrics, and visualization of results.

## Objectives
- Classify songs into genres using extracted audio features (e.g., MFCCs, spectral features).
- Compare traditional machine learning (Random Forest) with deep learning (CNNs).
- Evaluate the impact of transfer learning and spectrogram-based methods.
- Assess model performance using accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset
- **File**: `features_30_sec.csv`
- **Source**: GTZAN Genre Collection or similar (assumed to be stored in Google Drive as per the notebook).
- **Key Columns**:
  - Features: `chroma_stft_mean`, `chroma_stft_var`, `rms_mean`, `rms_var`, `spectral_centroid`, etc. (audio features).
  - Target: Genre label (e.g., rock, pop, jazz, etc.).
- **Size**: ~1,000 entries (typical for GTZAN; confirm with actual file).
- **Preprocessing**: Feature scaling with `StandardScaler`, label encoding for genres, train/test split. Spectrograms used for CNN input.

## Requirements
- Python 3.x (tested in Google Colab with GPU support; Python 3.12+ equivalent).
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow (for CNNs)
  - joblib
- Hardware: GPU recommended for CNN training (e.g., Colab T4).

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
   ```
3. Place `features_30_sec.csv` in the working directory or mount Google Drive with the dataset (as per the notebook).
4. For spectrogram-based CNN, ensure audio files or precomputed spectrograms are available.

## Usage
1. Open the Jupyter notebook: `Task 6 - Music Genre Classification Description.ipynb`.
2. Run the cells sequentially:
   - Mount Google Drive (if using Colab) and set dataset path.
   - Load and preprocess data (scale features, encode labels).
   - Train and evaluate Random Forest model.
   - Train and evaluate CNN model with spectrogram inputs.
   - Visualize confusion matrices and performance metrics.
3. Customize:
   - Adjust Random Forest hyperparameters (e.g., via GridSearchCV).
   - Modify CNN architecture or add transfer learning (e.g., VGG16).
   - Experiment with different audio features or spectrogram sizes.

Example command to start Jupyter (local):
```
jupyter notebook
```
Or use Google Colab for GPU support.

## Results
- **Random Forest**:
  - Accuracy: ~65-75% (depending on features and tuning).
  - F1-Score (macro avg): ~0.65.
- **CNN**:
  - Accuracy: ~75-85% (improved with spectrograms).
  - F1-Score (macro avg): ~0.75.
- **Key Insights**: CNNs outperform Random Forest due to their ability to capture patterns in spectrograms. Transfer learning (not fully implemented) could further boost performance.

## Visualization
- **Confusion Matrix**: Heatmap showing classification performance across genres for both models.
- **Feature Importance**: (For Random Forest) Top audio features contributing to predictions.

## Conclusion
CNNs provide superior performance for music genre classification when using spectrogram inputs, leveraging spatial patterns in audio data. Random Forest is effective for tabular feature data but less accurate. Transfer learning with pretrained models (e.g., VGG16) could enhance results but requires additional setup. Future improvements include exploring hybrid models or incorporating temporal features (e.g., via LSTMs).

## License
This project is for educational purposes. No specific license is applied.

## Author
Nazia Shar

## Contact
For questions or contributions, open an issue or pull request on GitHub.