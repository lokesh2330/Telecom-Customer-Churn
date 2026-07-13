import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs("Models",exist_ok=True)

#Load Dataset
df = pd.read_csv("Dataset/Telco_Customer_Churn.csv")

#Original DataSet Shape
print("-" * 50)
print("Original Dataset Shape")
print(df.shape)

# Remove Customer ID
df.drop("customerID", axis=1, inplace=True)
print("\ncustomerID Deleted")

# Convert TotalCharges to Numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

#Missing Values of TotalCharges
print("\nTotal Charges Missing Values:",end="0")
print(df["TotalCharges"].isnull().sum())

# Handle Missing Values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Encode Categorical Columns
label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(
    include=["object"]
).columns
    
# To save mapping values
mapping_list = []

for column in categorical_columns:

    le = LabelEncoder()
    le.fit(df[column])

    for category, code in zip(
        le.classes_,
        le.transform(le.classes_)
    ):
        mapping_list.append(
            [column, category, code]
        )

mapping_df = pd.DataFrame(
    mapping_list,
    columns=["Column", "Category", "Encoded_Value"]
)

mapping_df.to_csv(
    "Models/Feature_Mapping.csv",
    index=False
)
print("\nMapping Values saved Successfully")
    
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(
        df[column]
    )
    
#Churn Distribution
print("\nTarget Variable Distribution")
print("-"*50)
print(df["Churn"].value_counts())


# Save Processed Dataset
df.to_csv(
    "Dataset/Processed_telco_churn.csv",
    index=False
)

# Display Information of Processed dataset
print("\nProcessed Dataset Shape")
print("-" * 50)
print(df.shape)

print("\nProcessed Dataset Info")
print("-" * 50)
print(df.info())

#First few Records
print("\n First 5 Records")
print("-" * 50)
print(df.head())

print("\nProcessed dataset saved successfully!")
print("Location: Dataset/Processed_Customer_Churn.csv")