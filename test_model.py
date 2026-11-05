
from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
data = pd.read_csv(BASE_DIR / "dataset" / "cleaned_leads.csv")

X = data.drop(columns=["Converted"])
y = data["Converted"]

# Load Member 2's XGBoost model from JSON
model = XGBClassifier()
model.load_model(BASE_DIR / "models" / "best_lead_model.json")

# Load preprocessor
preprocessor = joblib.load(
    BASE_DIR / "models" / "preprocessor.pkl"
)

print("Dataset loaded:", X.shape)

# Apply preprocessing
X_processed = preprocessor.transform(X)

print("Preprocessing successful!")
print("Processed shape:", X_processed.shape)

# Check model
print("Model:", type(model).__name__)

# Predict
predictions = model.predict(X_processed)

print("Prediction successful!")
print("First 10 predictions:", predictions[:10])

# Conversion probabilities
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_processed)
    print("First 10 conversion probabilities:")
    print(probabilities[:10, 1])

print("MODEL INTEGRATION TEST COMPLETED!")
