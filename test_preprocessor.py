import joblib
import pandas as pd

# Load the preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_leads.csv")

# Remove target column
X = df.drop(columns=["Converted"])

# Transform the data
X_processed = preprocessor.transform(X)

print("Preprocessor loaded successfully!")
print("Original shape:", X.shape)
print("Processed shape:", X_processed.shape)