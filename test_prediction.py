import joblib
import pandas as pd
import xgboost as xgb

# Load dataset
df = pd.read_csv("dataset/cleaned_leads.csv")

# Separate features and target
X = df.drop(columns=["Converted"])

# Load preprocessor

preprocessor = joblib.load(
    "models/preprocessor_member2_final.pkl"
)
# Transform data
X_processed = preprocessor.transform(X)

# Load XGBoost model
model = xgb.XGBClassifier()
model.load_model("models/best_lead_model.json")

# Predict conversion probabilities
probabilities = model.predict_proba(X_processed)

# Display first 5 predictions
for i in range(5):
    print(
        f"Lead {i + 1}: "
        f"Conversion Probability = {probabilities[i][1] * 100:.2f}%"
    )