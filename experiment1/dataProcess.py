# ==========================================================
# TITANIC DATA PREPROCESSING
# Part 1
# ==========================================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ----------------------------------------------------
# 2. Load Dataset
# ----------------------------------------------------

df = pd.read_csv("train.csv")

print("========== BASIC INFORMATION ==========")
print(df.info())

print("\nShape of Dataset:", df.shape)

print("\nStatistical Summary")
print(df.describe())

# ----------------------------------------------------
# 3. Rows, Columns, Data Types, Missing Values
# ----------------------------------------------------

print("\nRows :", df.shape[0])
print("Columns :", df.shape[1])

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

# ----------------------------------------------------
# 4. Data Inspection
# ----------------------------------------------------

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nRandom 5 Rows")
print(df.sample(5))

# Save original dataset
original_df = df.copy()

# ----------------------------------------------------
# 5. Handle Missing Values
# ----------------------------------------------------

print("\nMissing Values Before")
print(df.isnull().sum())

# Fill Age using Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked using Mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin because of many missing values
df.drop(columns=["Cabin"], inplace=True)

print("\nMissing Values After")
print(df.isnull().sum())

# ----------------------------------------------------
# 6. Remove Duplicate Records
# ----------------------------------------------------

print("\nDuplicate Rows Before:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicate Rows After:", df.duplicated().sum())

# ----------------------------------------------------
# 7. Detect Outliers (Fare)
# ----------------------------------------------------

plt.figure(figsize=(7,4))
sns.boxplot(x=df["Fare"])
plt.title("Fare Before Outlier Handling")
plt.show()

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Fare"] < lower) | (df["Fare"] > upper)]

print("Number of Outliers:", len(outliers))

# ----------------------------------------------------
# 8. Handle Outliers (Capping)
# ----------------------------------------------------

df["Fare"] = np.where(df["Fare"] > upper, upper, df["Fare"])
df["Fare"] = np.where(df["Fare"] < lower, lower, df["Fare"])

plt.figure(figsize=(7,4))
sns.boxplot(x=df["Fare"])
plt.title("Fare After Outlier Handling")
plt.show()

# ==========================================================
# TITANIC DATA PREPROCESSING
# Part 2
# ==========================================================

# ----------------------------------------------------
# 9. Convert Categorical Variables
# Label Encoding
# ----------------------------------------------------

print("========== LABEL ENCODING ==========")

le = LabelEncoder()

# Encode categorical columns
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

print(df[["Sex", "Embarked"]].head())

# ----------------------------------------------------
# One-Hot Encoding
# ----------------------------------------------------

print("\n========== ONE HOT ENCODING ==========")

# Reload original dataset for one-hot encoding example
df_onehot = pd.read_csv("train.csv")

# Handle missing values
df_onehot["Age"] = df_onehot["Age"].fillna(df_onehot["Age"].median())
df_onehot["Embarked"] = df_onehot["Embarked"].fillna(df_onehot["Embarked"].mode()[0])

# Drop Cabin
df_onehot.drop(columns=["Cabin"], inplace=True)

# Apply One-Hot Encoding
df_onehot = pd.get_dummies(
    df_onehot,
    columns=["Sex", "Embarked"],
    drop_first=True
)

print(df_onehot.head())

# ----------------------------------------------------
# 10. Feature Scaling
# ----------------------------------------------------

print("\n========== FEATURE SCALING ==========")

# Min-Max Scaling
minmax = MinMaxScaler()

df["Age_MinMax"] = minmax.fit_transform(df[["Age"]])
df["Fare_MinMax"] = minmax.fit_transform(df[["Fare"]])

# Standardization (Z-score)
standard = StandardScaler()

df["Age_Standard"] = standard.fit_transform(df[["Age"]])
df["Fare_Standard"] = standard.fit_transform(df[["Fare"]])

print(df[[
    "Age",
    "Age_MinMax",
    "Age_Standard",
    "Fare",
    "Fare_MinMax",
    "Fare_Standard"
]].head())

# ----------------------------------------------------
# 11. Feature Selection using Correlation
# ----------------------------------------------------

print("\n========== CORRELATION WITH SURVIVED ==========")

corr = df.corr(numeric_only=True)

print(corr["Survived"].sort_values(ascending=False))

# ----------------------------------------------------
# 12. Correlation Matrix and Heatmap
# ----------------------------------------------------

plt.figure(figsize=(12,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------------------------
# 13. Feature Engineering
# ----------------------------------------------------

print("\n========== FEATURE ENGINEERING ==========")

# Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# IsAlone Feature
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

print(df[["SibSp", "Parch", "FamilySize", "IsAlone"]].head())

# ----------------------------------------------------
# Display New Features
# ----------------------------------------------------

print("\nDataset with New Features")
print(df.head())

# ==========================================================
# TITANIC DATA PREPROCESSING
# Part 3
# ==========================================================

# ----------------------------------------------------
# 14. Convert Data Types
# ----------------------------------------------------

print("========== DATA TYPE CONVERSION ==========")

# Convert PassengerId from int to float
df["PassengerId"] = df["PassengerId"].astype(float)

# Convert Pclass to category
df["Pclass"] = df["Pclass"].astype("category")

# Convert Survived to integer
df["Survived"] = df["Survived"].astype(int)

print(df.dtypes)

# ----------------------------------------------------
# 15. Handle Inconsistent and Noisy Data
# ----------------------------------------------------

print("\n========== DATA CLEANING ==========")

# Remove extra spaces from Name
df["Name"] = df["Name"].str.strip()

# Convert names to Title Case
df["Name"] = df["Name"].str.title()

# Remove duplicate spaces
df["Name"] = df["Name"].str.replace(r"\s+", " ", regex=True)

print(df["Name"].head())

# ----------------------------------------------------
# 16. Train-Test Split
# ----------------------------------------------------

print("\n========== TRAIN TEST SPLIT ==========")

# Drop non-numeric columns
X = df.drop(
    columns=[
        "Survived",
        "Name",
        "Ticket"
    ]
)

# Keep only numeric columns
X = X.select_dtypes(include=np.number)

y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ----------------------------------------------------
# 17. Data Balancing using SMOTE
# ----------------------------------------------------

print("\n========== BEFORE SMOTE ==========")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)

print("\n========== AFTER SMOTE ==========")
print(y_train_balanced.value_counts())

# ----------------------------------------------------
# 18. Distribution Before and After Scaling
# ----------------------------------------------------

# Age Before Scaling
plt.figure(figsize=(6,4))
sns.histplot(original_df["Age"].dropna(), kde=True)
plt.title("Age Before Preprocessing")
plt.xlabel("Age")
plt.show()

# Age After Min-Max Scaling
plt.figure(figsize=(6,4))
sns.histplot(df["Age_MinMax"], kde=True)
plt.title("Age After Min-Max Scaling")
plt.xlabel("Scaled Age")
plt.show()

# Fare Before Scaling
plt.figure(figsize=(6,4))
sns.histplot(original_df["Fare"], kde=True)
plt.title("Fare Before Preprocessing")
plt.xlabel("Fare")
plt.show()

# Fare After Standardization
plt.figure(figsize=(6,4))
sns.histplot(df["Fare_Standard"], kde=True)
plt.title("Fare After Standardization")
plt.xlabel("Standardized Fare")
plt.show()

print("\nDistribution plots generated successfully.")