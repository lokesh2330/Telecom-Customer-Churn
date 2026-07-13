import pandas as pd
import joblib

# Load Trained Model
model = joblib.load(
    "Models/churn_model.pkl"
)

print("MODEL LOADED SUCCESSFULLY")

# Sample Customer Data
sample_customer = pd.DataFrame([{
    "gender":1,
    "SeniorCitizen":0,
    "Partner":1,
    "Dependents":0,
    "tenure":102,
    "InternetService":1,
    "Contract":0,
    "PaymentMethod":2,
    "MonthlyCharges":85.50,
    "TotalCharges":1260.00,
    "OnlineSecurity":0,
    "TechSupport":0,
    "OnlineBackup":2
}])


# Prediction
prediction = model.predict(sample_customer)[0]

probability = model.predict_proba(
    sample_customer
)[0][1]


# Result
print("\nPrediction Result")
print("-" * 50)

if prediction == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")

print(
    f"\nChurn Probability : {probability:.2%}"
)