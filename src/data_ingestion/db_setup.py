import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Use SQLite by default for easy local execution, but allow PostgreSQL via env
DB_URL = os.getenv('DATABASE_URL', 'sqlite:///../../data/processed/churn_db.sqlite')

def setup_database():
    engine = create_engine(DB_URL)
    print(f"Connected to database: {DB_URL}")
    
    # Load augmented data
    data_path = '../../data/processed/telco_churn_augmented.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Run EDA script first.")
        
    df = pd.read_csv(data_path)
    
    # Basic cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True) # Fill missing with 0 for now
    
    # Split data into normalized tables (simulating CRM schema)
    
    # 1. Customers Table
    customers = df[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Churn']].copy()
    customers.to_sql('customers', engine, if_exists='replace', index=False)
    print("Created 'customers' table.")
    
    # 2. Subscriptions / Services Table
    services = df[['customerID', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']].copy()
    services.to_sql('services', engine, if_exists='replace', index=False)
    print("Created 'services' table.")
    
    # 3. Billing Table
    billing = df[['customerID', 'MonthlyCharges', 'TotalCharges']].copy()
    billing.to_sql('billing', engine, if_exists='replace', index=False)
    print("Created 'billing' table.")
    
    # Verify tables
    with engine.connect() as conn:
        if DB_URL.startswith('sqlite'):
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result]
        else:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
            tables = [row[0] for row in result]
            
    print("Tables in database:", tables)

if __name__ == "__main__":
    setup_database()
