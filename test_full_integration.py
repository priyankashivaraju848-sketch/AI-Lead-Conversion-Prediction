import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import joblib
import pandas as pd
import xgboost as xgb

# Add Member 3 AI module path
AI_DIR = Path("src/Member3_AI_Recommendation")
sys.path.insert(0, str(AI_DIR.resolve()))

from main_ai import run_lead_ai

# Load dataset
df = pd.read_csv("dataset/cleaned_leads.csv")

# Select one lead
lead = df.drop(columns=["Converted"]).iloc[[0]]

# Load preprocessing pipeline
preprocessor = joblib.load("models/preprocessor.pkl")

# Transform lead
X_processed = preprocessor.transform(lead)

# Load ML model
model = xgb.XGBClassifier()
model.load_model("models/best_lead_model.json")

# Predict conversion probability
probability = float(model.predict_proba(X_processed)[0][1])

# Convert lead details into dictionary
lead_details = lead.iloc[0].to_dict()

print("\n========== ML PREDICTION ==========")
print(f"Conversion Probability: {probability * 100:.2f}%")

# Generate AI outputs
print("\n========== AI RECOMMENDATIONS ==========")

result = run_lead_ai(lead_details, probability)

print("\nLead Category:", result["category"])
print("\nAI Explanation:")
print(result["explanation"])

print("\nSales Recommendation:")
print(result["recommendation"])

print("\nFollow-up Strategy:")
print(result["follow_up"])