Machine Learning Project utilizing TensorFlow to determine Body Fat Percentage

This project uses machine learning to predict body fat percentage based on various physical attributes. 
It employs a neural network model built with TensorFlow/Keras and features a complete pipeline from data preprocessing to model evaluation.

Features:
- Data Preprocessing: Filters outliers using the Interquartile Range (IQR) method and normalizes the dataset for better model performance.
- Neural Network Model: A custom regression model designed and trained to predict body fat percentage.
- Performance Metrics: Evaluates the model using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.
- Visualization: (Optional) Provides insights into the model’s predictions and performance using visualizations.

Dataset:
- Body Fat Percentage
- Age
- Weight
- Height
- Density
- Abdomen Circumference

Findings:
In research, the hybrid model such as "the MR-SVR achieved the best performance with an RMSE of 4.6427". 
In comparison, the refined model, which relies heavily upon DEEP preprocessing techniques and neural networks, obtained an RMSE of 2.8344.

Data Source:
https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset

Research Documntation Comparison:
- https://www.mdpi.com/2076-3417/11/21/9797

Citation:
- Hussain, S. A., Cavus, N., & Sekeroglu, B. (2021). Hybrid Machine Learning Model for Body Fat Percentage Prediction Based on Support Vector Regression and Emotional Artificial Neural Networks. Applied Sciences, 11(21), 9797. https://doi.org/10.3390/app11219797

Neural Network Model Performance:
- MAE: 2.227394756616331
- MSE: 8.034292852539881
- R2: 0.8942721848268207
- RMSE: 2.8344828192352622
