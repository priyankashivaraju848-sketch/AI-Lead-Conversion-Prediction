import joblib
import xgboost as xgb

# Load test data
X_test = joblib.load("models/X_test_processed.pkl")

# Load the JSON model
model = xgb.XGBClassifier()
model.load_model("models/best_lead_model.json")

# Generate predictions
probabilities = model.predict_proba(X_test)[:, 1]

print("JSON model loaded successfully!")
print("Number of features:", model.n_features_in_)
print("Number of predictions:", len(probabilities))
print("First 5 probabilities:", probabilities[:5])