# Bank-Fraud-Detection

Overview

This project aims to build a machine learning model to detect fraudulent transactions in banking systems. The model leverages classification techniques to distinguish between legitimate and fraudulent transactions based on historical data.

# Dataset Information

Dataset Name: ["Bank Transaction Fraud Dataset"](https://www.kaggle.com/datasets/orangelmendez/bank-fraud)

Number of Records: 200,000 rows

Number of Features: 19 Columns

Target Variable: is_fraud (1 - Fraudulent, 0 - Non-Fraudulent)

**NOTE:** The data is pulled directly from kaggle and ran in a google colab environment. If you plan to run the data locally, download the dataset from the link above.

**Feature Description**

The original dataset came from ["Original Dataset"](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection). Although this dataset contained sensitive information that is not relevant to this model. In an attempt to protect this information, I dropped the corresponding features, and uploaded an updated version of the dataset.

# EDA

**Modeling Steps**

1. Data Preprocessing

I ensured that this dataset did not have any missing values. Once confirmed, I noticed that the dataset is heavily imbalanced, (95% non-fraud, 5% fraud).  Converting categorical data into numerical values using one-hot encoding. After encoding, normalizing and creating some synthetic data to balance the dataset. I performed an  80/20 split between the training and testing sets.

2. Model Selection

I wanted to test various machine learning models to find the best fit for fraud detection. I started with a baseline Logistic Regression model to start for classification. Using XGBoost for gradient boosting to handle imbalance within the dataset well, then using RandomForest classifier to assemble learning method that improves predictive accuracy. Neural Network as a deep learning approaches that capture complex relationships in the data. Finally, cross-validation to select the best model.

3. Model Training & Optimization 
Tuning hyperparameters using GridSearchCV or Bayesian Optimization to find the best settings. Fraudulent transactions are often rare compared to legitimate ones. Addressing the class imbalance, using SMOTE (Synthetic Minority Over-sampling Technique), undersampling, or weighted loss functions can balance the dataset.

5. Model Evaluation

Using metrics Accuracy, Precision, Recall, F1-score. Including ROC-AUC as well, I generate confusion matrices and classification reports.

## Results & Findings

![Model metrics](https://github.com/omendez930/Bank-Fraud-Detection/blob/main/Photos/Picture1.png)

## Best-performing model: 

![Confusion Matrix](https://github.com/omendez930/Bank-Fraud-Detection/blob/main/Photos/output.png)

## Key insights from the data analysis

* Recall score is 68% as of now. There is a trade-off with the precision metric of 5%. This will lead to misclassification of transactions that are legitimate but flagged as fraudulent.


## Project Links

Model Notebook: ["Jupyter Notebook"](https://github.com/omendez930/Bank-Fraud-Detection/blob/main/notebook.ipynb)

Tableau VIsualizations: ['Bank Fraud Detection Visualizations'](https://public.tableau.com/app/profile/orangel.mendez/viz/Bank_Fraud_Detection/Dashboard1?publish=yes)

Presentation/Report: [Link to any slides or reports](https://github.com/omendez930/Bank-Fraud-Detection/blob/main/Bank%20Fraud%20Detection.pdf)

## Future Work

Incorporate Pyspark, and Neural Networks.

Improve model performance with ensemble methods

Deploy model using Flask or FastAPI

Integrate real-time fraud detection in banking systems

## **Author**

Orangel Mendez
Email: ["omendez30@gmail.com"](omendez30@gmail.com)
Github profile: ["GitHub Profile"](https://github.com/omendez930)

