import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#To create folder for Images
os.makedirs("Images/Analysis",exist_ok=True)

#Load Dataset
df=pd.read_csv("Dataset/Telco_Customer_Churn.csv")
#Dataset Information

#Shape
print("\nDataset Shape")
print("-"*50)
print(df.shape)

#First few rows
print("\nFirst 5 rows")
print("-"*50)
print(df.head())

#Missing Values
print("\nMissing Values")
print("-"*50)
print(df.isnull().sum())

#Duplicate Values
print("\nDuplicated IDs")
print("-"*50)
print(df["customerID"].duplicated().sum())

#DataTypes
print("\nDataTypes:")
print("-"*50)
print(df.info())

#Summary of numerical datatypes
print("\nStatistical Summary")
print("-" * 50)
print(df.describe())

# --------------------------------------------------
# Target Variable Analysis
# --------------------------------------------------
print("\nChurn Distribution")
print("-" * 50)
print(df["Churn"].value_counts())

#Figure of Churn Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Churn")
plt.title("Churn Distribution")
plt.savefig("Images/Analysis/churn_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Churn")
plt.savefig("Images/Analysis/tenure_vs_Churn.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.savefig("Images/Analysis/monthly_charges_vs_Churn.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.savefig("Images/Analysis/contract_vs_Churn.png")
plt.close()

#Additional 

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="SeniorCitizen", hue="Churn")
plt.title("Senior Citizen vs Churn")
plt.savefig("Images/Analysis/Seniority_vs_Churn.png")
plt.close()

plt.figure(figsize=(8,4))
sns.countplot(data=df,x="Partner",hue="Churn")
plt.title("Partner vs Churn")
plt.savefig("Images/Analysis/Partner_vs_Churn.png")
plt.close()


plt.figure(figsize=(8,4))
sns.countplot(data=df,x="InternetService",hue="Churn")
plt.title("InternetService vs Churn")
plt.savefig("Images/Analysis/InternetService_vs_Churn.png")
plt.close()


#Message
print("\nEDA Completed Successfully")
print("Graphs saved in Images/Analysis folder")