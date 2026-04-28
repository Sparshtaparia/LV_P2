# Exploratory Data Analysis (EDA) Report

## 1. Dataset Overview
The dataset selected for this project is the Telco Customer Churn dataset, combined with synthetic data augmentation using `Faker` to scale the volume to 10,000 records.

- **Original Dataset**: 7,043 rows, 21 columns
- **Augmented Dataset**: 10,000 rows, 21 columns
- **Features**: `customerID`, `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `StreamingTV`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`.

## 2. Missing Values Analysis
A quick check on the original dataset revealed that `TotalCharges` contained 11 missing values (empty strings). These were coerced to `NaN` and will be imputed during the feature engineering pipeline. All other features have complete data.

## 3. Class Balance (Churn)
The initial dataset has a natural class imbalance:
- **No Churn**: ~73%
- **Churn (Yes)**: ~27%

*See `class_balance.png` in the EDA plots directory.*
This imbalance will be handled later using SMOTE oversampling and class weights.

## 4. Feature Distributions
- **Tenure**: A bimodal distribution, showing many customers leave within the first few months, while another large group remains loyal for over 60 months.
- **Monthly Charges**: Customers who churn tend to have higher monthly charges, typically clustering between $70 and $100.
- **Total Charges**: Skewed to the right, mostly because newer customers haven't accumulated high charges yet.

*See `dist_tenure.png`, `dist_MonthlyCharges.png`, and `dist_TotalCharges.png` in the EDA plots directory.*

## 5. Correlation Analysis
A correlation heatmap between numeric variables and the binary `Churn` target indicates:
- **Tenure** has a strong negative correlation with churn (longer tenure = less likely to churn).
- **Monthly Charges** have a positive correlation with churn (higher charges = more likely to churn).
- **Total Charges** are highly correlated with tenure, which is expected.

*See `correlation_heatmap.png` in the EDA plots directory.*

## 6. Synthetic Augmentation
To mimic a larger enterprise environment and ensure robustness, `Faker` was used to generate 2,957 additional synthetic customer records following the schema of the original dataset. These were concatenated to reach exactly 10,000 records.

## Conclusion & Next Steps
The dataset is now cleaned and augmented. The next steps involve setting up a PostgreSQL database to house the data properly, followed by advanced feature engineering (RFM scoring, binning) and training a baseline Random Forest classifier.
