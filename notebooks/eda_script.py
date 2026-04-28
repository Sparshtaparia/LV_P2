import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from faker import Faker

# Configure matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')

# Create a directory to save plots
os.makedirs('../docs/eda_plots', exist_ok=True)

# Load the dataset
data_path = '../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("\n--- Missing Values ---")
# TotalCharges is a string, let's convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df.isnull().sum())

# Basic info
print("\n--- Data Info ---")
print(df.info())

# Class balance
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Class Balance: Churn')
plt.savefig('../docs/eda_plots/class_balance.png')
plt.close()
print("\nClass Balance saved to docs/eda_plots/class_balance.png")

# Feature distributions (Numeric)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=col, hue='Churn', kde=True, palette='Set2', element="step")
    plt.title(f'Distribution of {col} by Churn')
    plt.savefig(f'../docs/eda_plots/dist_{col}.png')
    plt.close()
print("Numeric feature distributions saved.")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
# Convert Churn to binary for correlation
df_corr = df.copy()
df_corr['Churn'] = df_corr['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
corr_matrix = df_corr[numeric_cols + ['Churn']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('../docs/eda_plots/correlation_heatmap.png')
plt.close()
print("Correlation heatmap saved.")

# Synthetic Augmentation using Faker to double the dataset size
print("\n--- Synthetic Augmentation ---")
fake = Faker()

def generate_synthetic_data(n=2000):
    synthetic_data = []
    for _ in range(n):
        customer = {
            'customerID': fake.unique.uuid4()[:10],
            'gender': fake.random_element(elements=('Male', 'Female')),
            'SeniorCitizen': fake.random_element(elements=(0, 1)),
            'Partner': fake.random_element(elements=('Yes', 'No')),
            'Dependents': fake.random_element(elements=('Yes', 'No')),
            'tenure': fake.random_int(min=0, max=72),
            'PhoneService': fake.random_element(elements=('Yes', 'No')),
            'MultipleLines': fake.random_element(elements=('Yes', 'No', 'No phone service')),
            'InternetService': fake.random_element(elements=('DSL', 'Fiber optic', 'No')),
            'OnlineSecurity': fake.random_element(elements=('Yes', 'No', 'No internet service')),
            'OnlineBackup': fake.random_element(elements=('Yes', 'No', 'No internet service')),
            'DeviceProtection': fake.random_element(elements=('Yes', 'No', 'No internet service')),
            'TechSupport': fake.random_element(elements=('Yes', 'No', 'No internet service')),
            'StreamingTV': fake.random_element(elements=('Yes', 'No', 'No internet service')),
            'StreamingMovies': fake.random_element(elements=('Yes', 'No', 'No internet service')),
            'Contract': fake.random_element(elements=('Month-to-month', 'One year', 'Two year')),
            'PaperlessBilling': fake.random_element(elements=('Yes', 'No')),
            'PaymentMethod': fake.random_element(elements=('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')),
            'MonthlyCharges': round(fake.random.uniform(18.25, 118.75), 2),
            'Churn': fake.random_element(elements=('Yes', 'No'))
        }
        customer['TotalCharges'] = round(customer['MonthlyCharges'] * customer['tenure'], 2)
        synthetic_data.append(customer)
    return pd.DataFrame(synthetic_data)

synthetic_df = generate_synthetic_data(2957) # Pad to make exactly 10,000 total rows
print(f"Generated {synthetic_df.shape[0]} synthetic records.")

# Combine
combined_df = pd.concat([df, synthetic_df], ignore_index=True)
print("Combined Dataset Shape:", combined_df.shape)

os.makedirs('../data/processed', exist_ok=True)
combined_df.to_csv('../data/processed/telco_churn_augmented.csv', index=False)
print("Augmented dataset saved to data/processed/telco_churn_augmented.csv")

