import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

os.makedirs("Images/Metrics",exist_ok=True)

# Load Processed Dataset
df = pd.read_csv("Dataset/Processed_telco_churn.csv")

#Dataset Shape
print("Dataset Shape")
print("-" * 50)
print(df.shape)

# Features and Target
selected_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "OnlineSecurity",
    "TechSupport",
    "OnlineBackup"
]
X = df[selected_features]
y = df["Churn"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)
print("\nModel Trained Successfully!")

# Cross Validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores")
print(cv_scores)

print(
    "\nAverage Cross Validation Accuracy:",
    round(cv_scores.mean(), 4)
)

# Predictions
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)


# Print Metrics
print("\nMODEL EVALUATION")
print("-" * 50)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)

# Save Confusion Matrix Image
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig(
    "Images/Metrics/confusion_matrix.png",
    bbox_inches="tight"
)
plt.close()


# Save ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(
    "Images/Metrics/roc_curve.png",
    bbox_inches="tight"
)

plt.close()

# Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print("-" * 50)
print(importance_df.head(19))


# Save Feature Importance CSV
importance_df.to_csv(
    "Models/Feature_Importance.csv",
    index=False
)


# Save Feature Importance Graph
top_features = importance_df.head(10)
plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig(
    "Images/Metrics/feature_importance.png",
    bbox_inches="tight"
)
plt.close()


# Save Model

joblib.dump(
    model,
    "Models/churn_model.pkl"
)

print("\nModel Saved Successfully!")
print("Location : Models/churn_model.pkl")

print("\nFeature Importance Saved Successfully!")
print("Location : Models/Feature_Importance.csv")

print("\nImages/Metrics Saved Successfully!")
print("Location : Images/Metrics")



#print(list(model.feature_names_in_))