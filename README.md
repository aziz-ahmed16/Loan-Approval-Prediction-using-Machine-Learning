# Loan Approval Prediction using Machine Learning
## Overview
This project presents a Loan Approval Prediction System built with Python and machine learning. The aim is to predict whether or not a loan application will be approved, based on historical application data. Such predictive modeling is widely used in fintech and banking to automate and optimize decision-making processes.
Features
- Cleans and preprocesses real-world loan application data
- Interactive Exploratory Data Analysis (EDA) with Plotly visualizations
- Thoughtful handling of missing values and outliers in both categorical and numerical features
- One-hot encoding of categorical features
- Standardization of numerical features
- Trains a Support Vector Machine (SVM) classifier for binary classification
- Evaluates model performance using accuracy, confusion matrix, and classification report
- Appends predictions to the test data for further analysis
## Dataset
This project uses the publicly available Loan Prediction dataset, sourced from Kaggle.
- Source: [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)

You may download the dataset from Kaggle or from the repository and save it as loan_prediction.csv in your working directory to reproduce the results of this project.

## Project Workflow
### 1. Data Loading & Cleaning

- Loads the CSV dataset
- Removes unnecessary identifier columns (like Loan_ID)
- Fills missing categorical values with the mode (most frequent value)
- Fills missing numerical values with the median or mode, as appropriate 

### 2. Exploratory Data Analysis (EDA)

- Visualizes loan approval status, gender, marital status, education, self-employment, income, property area, and credit history distributions with Plotly
- Uses box plots and histograms to inspect relationships and outliers
- Removes outliers from income features using the IQR method

### 3. Feature Engineering

- One-hot encodes relevant categorical columns
- Splits the dataset into features (X) and target (y)
- Splits the data into training and testing sets (80% train, 20% test)

### 4. Data Scaling

- Standardizes the numerical columns (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History) using StandardScaler for optimal SVM performance

### 5. Model Training

- Trains a Support Vector Machine (SVC) classifier on the training data

### 6. Model Evaluation

- Makes predictions on the test set
- Evaluates performance using:
  * Accuracy
  * Confusion Matrix
  * Classification Report

### 7. Results Analysis

- Outputs the predictions alongside actual values in the test set for further review


## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- plotly

## Installation:
```bash
 pip install pandas numpy scikit-learn plotly 
 ```

## Usage
1. Clone this repository.
2. Download the dataset from Kaggle and place loan_prediction.csv in the project directory.
3. Open Loan_Prediction.ipynb and run all cells sequentially.
4. Review the data analysis, model predictions, and EDA visualizations in the notebook.

## Results
- The notebook outputs:
  * Test accuracy
  * Confusion matrix
  * Classification report

- You may further improve results by trying different models, feature engineering, or hyperparameter tuning.
