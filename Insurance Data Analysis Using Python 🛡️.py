
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

file_path = Path(__file__).parent / "data" / "insurance_data.csv"
df = pd.read_csv("insurance_data.csv")

df.columns = df.columns.str.strip()

required_columns = [
    "Customer_ID", "Age", "Gender", "BMI",
    "Children", "Smoker", "Region", "Insurance_Charges"
]

if not all(col in df.columns for col in required_columns):
    print("Error: Check your CSV column names.")
    print("Available columns:", df.columns.tolist())
    raise SystemExit

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nTotal Customers:", len(df))
print("Average Age:", round(np.mean(df["Age"]), 2))
print("Average BMI:", round(np.mean(df["BMI"]), 2))
print(
    "Average Insurance Charges:",
    round(np.mean(df["Insurance_Charges"]), 2)
)
print("Maximum Insurance Charges:", df["Insurance_Charges"].max())
print("Minimum Insurance Charges:", df["Insurance_Charges"].min())

print("\nCustomers by Smoking Status:")
print(df["Smoker"].value_counts())

print("\nAverage Charges by Smoking Status:")
print(df.groupby("Smoker")["Insurance_Charges"].mean().round(2))

print("\nAverage Charges by Region:")
print(df.groupby("Region")["Insurance_Charges"].mean().round(2))

print("\nAverage Charges by Gender:")
print(df.groupby("Gender")["Insurance_Charges"].mean().round(2))

sns.set_theme(style="whitegrid")

sns.barplot(data=df, x="Smoker", y="Insurance_Charges")
plt.title("Insurance Charges by Smoking Status")
plt.tight_layout()
plt.show()

sns.scatterplot(
    data=df,
    x="Age",
    y="Insurance_Charges",
    hue="Smoker"
)
plt.title("Age vs Insurance Charges")
plt.tight_layout()
plt.show()

sns.scatterplot(
    data=df,
    x="BMI",
    y="Insurance_Charges",
    hue="Smoker"
)
plt.title("BMI vs Insurance Charges")
plt.tight_layout()
plt.show()

sns.barplot(data=df, x="Region", y="Insurance_Charges")
plt.title("Insurance Charges by Region")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

sns.histplot(data=df, x="Insurance_Charges", bins=10, kde=True)
plt.title("Distribution of Insurance Charges")
plt.tight_layout()
plt.show()

print("\nInsurance Data Analysis Completed!")
