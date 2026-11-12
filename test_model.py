import xgboost as xgb

model = xgb.XGBClassifier()

model.load_model("models/best_lead_model.json")

print("XGBoost model loaded successfully!")
print("Number of features:", model.n_features_in_)
print("Classes:", model.classes_)