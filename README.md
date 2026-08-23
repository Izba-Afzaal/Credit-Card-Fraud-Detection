💳 Credit Card Fraud Detection
📌 Project Overview

This project uses Machine Learning to detect fraudulent credit card transactions. The goal is to classify transactions into two categories:

0 → Genuine Transaction
1 → Fraudulent Transaction

The project includes data analysis, data visualization, feature scaling, machine learning model training, handling imbalanced data, model evaluation, fraud prediction, and an interactive Streamlit dashboard.

📊 Dataset

The dataset contains 284,807 credit card transactions with 31 columns.

Genuine Transactions: 284,315
Fraudulent Transactions: 492
Fraud Percentage: 0.17%

The dataset is highly imbalanced, making fraud detection a challenging machine learning problem.

Features

The dataset contains:

Time
V1 to V28
Amount
Class

The features V1 to V28 are anonymized features created using PCA (Principal Component Analysis).

The Class column represents:

0 → Genuine Transaction
1 → Fraudulent Transaction
🔍 Exploratory Data Analysis

The following analysis and visualizations were performed:

Genuine vs Fraudulent Transaction Count
Fraud Percentage Analysis
Average Transaction Amount Comparison
Transaction Amount Distribution

These visualizations help understand the highly imbalanced nature of the dataset.

⚙️ Data Preprocessing

The following preprocessing steps were performed:

Checked for missing values
Separated features and target variable
Split the dataset into training and testing sets
Used stratified sampling to preserve class distribution
Applied StandardScaler for feature scaling

Training data:

227,845 transactions

Testing data:

56,962 transactions

🤖 Machine Learning Models
1. Logistic Regression

A Logistic Regression model was trained using the original imbalanced dataset.

Model Performance
Metric	Score
Accuracy	99.91%
Precision	82.67%
Recall	63.27%
F1 Score	71.68%
[[56851    13]
 [   36    62]]
 2. Balanced Logistic Regression

Because the dataset contains very few fraudulent transactions, another Logistic Regression model was trained using:

class_weight="balanced"
Balanced Model Performance
Metric	Score
Accuracy	97.55%
Precision	6.10%
Recall	91.84%
F1 Score	11.44%

The balanced model significantly improved fraud detection recall, meaning it successfully detected a larger number of fraudulent transactions.
📈 Model Comparison
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	99.91%	82.67%	63.27%	71.68%
Balanced Logistic Regression	97.55%	6.10%	91.84%	11.44%
Key Insight

The original Logistic Regression model provides better overall precision and F1 score.

The balanced model achieves a much higher recall, which is important in fraud detection because detecting fraudulent transactions can be more valuable than maintaining high overall accuracy.

🖥️ Streamlit Dashboard

An interactive Streamlit dashboard was developed to demonstrate the fraud detection system.

The dashboard includes:

💳 Credit Card Fraud Detection Interface
💰 Transaction Input
📊 Fraud/Genuine Prediction
🚨 Fraud Alert
📈 Prediction Probability
📋 Model Performance Information
🎨 Professional User Interface
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Joblib
Streamlit
📂 Project Structure
Credit-Card-Fraud-Detection/
│
├── fraud_detection.py
├── app.py
├── fraud_model.pkl
├── fraud_scaler.pkl
├── README.md
└── requirements.txt
🚀 How to Run the Project
1. Clone the repository
git clone YOUR_REPOSITORY_LINK
2. Navigate to the project folder
cd Credit-Card-Fraud-Detection
3. Install the required libraries
pip install -r requirements.txt
4. Run the Streamlit application
python -m streamlit run app.py
🎯 Project Workflow
Dataset
   ↓
Data Understanding
   ↓
Data Analysis & Visualization
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Imbalanced Data Handling
   ↓
Fraud Prediction
   ↓
Model Saving
   ↓
Streamlit Dashboard
🔮 Future Improvements

Possible future improvements include:

Implementing Random Forest and XGBoost models
Hyperparameter tuning
Using SMOTE for advanced imbalance handling
Improving fraud prediction performance
Deploying the Streamlit application online
Adding additional interactive visualizations
👩‍💻 Author

CodeVertex-max

⭐ Conclusion

This project demonstrates an end-to-end Machine Learning workflow for detecting fraudulent credit card transactions. It includes data preprocessing, exploratory data analysis, handling class imbalance, model training, performance evaluation, model saving, and an interactive Streamlit dashboard.