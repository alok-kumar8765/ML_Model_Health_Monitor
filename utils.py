import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def generate_sample_data():
    """Generates a sample dataset and trains a baseline model if they don't exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "Age": np.random.randint(18, 70, size=n_samples),
        "Income": np.random.randint(20000, 150000, size=n_samples),
        "Purchase_Freq": np.random.randint(1, 30, size=n_samples),
        "Time_on_Site": np.random.randint(10, 1200, size=n_samples),
        "Total_Spend": np.random.randint(500, 50000, size=n_samples),
        "Location": np.random.choice([0, 1, 2, 3], size=n_samples)
    }
    
    df = pd.DataFrame(data)
    # Target: 0 for Normal, 1 for Anomaly (~13% anomalies)
    df["Target"] = np.random.choice([0, 1], size=n_samples, p=[0.868, 0.132])
    
    csv_path = "data/sample_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Train and save a model
    X = df.drop(columns=["Target"])
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "models/model.pkl")
    return df

def load_data(file):
    """Loads CSV or Excel data into a pandas DataFrame."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def get_recent_anomalies_df():
    """Returns sample recent anomaly records for the dashboard table."""
    return pd.DataFrame({
        "ID": [1001, 1002, 1003, 1004, 1005],
        "Date": ["2026-10-21", "2026-10-21", "2026-10-20", "2026-10-20", "2026-10-19"],
        "Feature": ["Total_Spend", "Purchase_Freq", "Age", "Income", "Time_on_Site"],
        "Actual Value": [52400, 35, 12, 950000, 4800],
        "Expected Range": ["500 - 20000", "1 - 10", "18 - 70", "20000 - 300000", "10 - 1200"],
        "Anomaly Score": [0.95, 0.91, 0.89, 0.87, 0.84],
        "Status": ["High", "High", "High", "High", "High"]
    })