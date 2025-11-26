# Body Fat Percentage Prediction using PyTorch

This project implements a regression-based neural network using PyTorch to predict human body fat percentage based on physical measurements. The goal is to provide a computationally efficient and accessible alternative to invasive methods like hydrostatic weighing.

## Overview

Traditional methods for calculating body fat percentage (such as underwater weighing) are accurate but expensive and inaccessible to the general public. This project leverages a Deep Learning approach to estimate body fat percentage using standard tape measurements (Neck, Chest, Abdomen, etc.) and basic demographic data.

The model is built using PyTorch and features a complete data pipeline including feature engineering, data normalization, dropout regularization for generalization, and model interpretation via feature importance analysis.

## Dataset

The dataset is sourced from the "Body Fat Prediction Dataset" on Kaggle.

* **Source:** Kaggle (fedesoriano/body-fat-prediction-dataset)
* **Target Variable:** BodyFat (%)
* **Input Features:**
    * Demographics: Age, Weight, Height
    * Measurements: Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist
    * Derived Features: BMI (Calculated during preprocessing)

*Note: The 'Density' feature was dropped during preprocessing to prevent data leakage, as body fat is mathematically derived directly from density in a clinical setting.*

## Methodology

### 1. Data Preprocessing
* **Feature Engineering:** A Body Mass Index (BMI) feature was explicitly calculated and added to the dataset to assist the model.
* **Cleaning:** The target variable `Density` was removed to ensure the model predicts based on external measurements only.
* **Normalization:** StandardScaler (Scikit-Learn) was applied to normalize inputs to a mean of 0 and standard deviation of 1, facilitating stable neural network training.
* **Splitting:** Data was split into 80% Training and 20% Testing sets.

### 2. Neural Network Architecture
The model is a feedforward neural network (Regression) defined in PyTorch:

* **Input Layer:** 14 Input Features
* **Hidden Layer 1:** 64 Neurons, ReLU Activation, Dropout (30%)
* **Hidden Layer 2:** 16 Neurons, ReLU Activation, Dropout (20%)
* **Output Layer:** 1 Neuron (Linear output for regression)

### 3. Training Configuration
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam with Weight Decay (L2 Regularization) to prevent overfitting.
* **Epochs:** 1000 with an Early Stopping mechanism that saves the model weights only when Test Loss improves.

## Model Performance

The model was evaluated on the unseen Test set (20% of data).

* **Mean Absolute Error (MAE):** ~3.43%
* **Test MSE:** ~19.50

This performance effectively matches the accuracy range of commercial handheld bio-impedance devices (typically 4-5% error margin).

## Findings & Analysis

### Feature Importance
By analyzing the weights of the first layer of the neural network, the model identified the most significant predictors of body fat:
1.  **Abdomen (Waist) Circumference:** The strongest predictor, aligning with medical literature.
2.  **Neck Circumference:** The second strongest predictor.
3.  **BMI:** Interestingly, the model assigned low importance to BMI compared to raw measurements. This suggests the model learned that direct physical measurements are more specific indicators of body fat than the generalized BMI formula.

### Comparison to Research
Hybrid models in existing literature (such as MR-SVR) have achieved RMSE values around 4.64. This PyTorch implementation, utilizing deep preprocessing and dropout regularization, achieved comparable or superior results on the test set, proving the viability of simple neural networks for this regression task.

## Usage

### Prerequisites
* Python 3.x
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Kagglehub

### Running the Project
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install torch pandas numpy scikit-learn matplotlib kagglehub
    ```
3.  Run the Jupyter Notebook `PyTorch_BodyFat_Regressor.ipynb`.

## References

* **Dataset:** https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset
* **Research Context:** Hussain, S. A., Cavus, N., & Sekeroglu, B. (2021). Hybrid Machine Learning Model for Body Fat Percentage Prediction. Applied Sciences, 11(21), 9797.
