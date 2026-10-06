import pandas as pd
import numpy as np

# ==========================================
# TASK 1 - DATA IMMERSION & WRANGLING
# ApexPlanet Data Analytics Internship
# ==========================================

# 1. LOAD DATASET
file_path = "ApexPlanet_DataAnalytics_Dataset.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# 2. BASIC DATA INSPECTION
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all"))

# 3. CHECK MISSING VALUES
print("\nMissing Values:")
print(df.isnull().sum())

# 4. CHECK EXACT DUPLICATES
duplicate_count = df.duplicated().sum()

print("\nExact duplicate rows:", duplicate_count)

# 5. CHECK DATA TYPES
print("\nData Types:")
print(df.dtypes)

# 6. CONVERT DATE COLUMN
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

# 7. HANDLE MISSING AGE
age_median = df["Age"].median()

df["Age_Imputed"] = df["Age"].isna().astype(int)

df["Age"] = df["Age"].fillna(age_median)

print("\nMissing Age after cleaning:",
      df["Age"].isnull().sum())

# 8. HANDLE MISSING CITY
df["City_Missing"] = df["City"].isna().astype(int)

df["City"] = df["City"].fillna("Unknown")

print("Missing City after cleaning:",
      df["City"].isnull().sum())

# 9. CREATE UNIQUE ROW IDENTIFIER
df["Order_Line_ID"] = range(1, len(df) + 1)

# 10. FEATURE ENGINEERING - DATE
df["Year"] = df["Order_Date"].dt.year

df["Month"] = df["Order_Date"].dt.month

df["Month_Name"] = df["Order_Date"].dt.month_name()

df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# 11. AGE GROUP
def age_group(age):

    if age < 18:
        return "Under 18"

    elif age <= 25:
        return "18-25"

    elif age <= 35:
        return "26-35"

    elif age <= 45:
        return "36-45"

    elif age <= 60:
        return "46-60"

    else:
        return "60+"

df["Age_Group"] = df["Age"].apply(age_group)

# 12. SALES VALIDATION
df["Calculated_Sales"] = (
    df["Quantity"] * df["Unit_Price"]
)

df["Sales_Check"] = np.isclose(
    df["Total_Sales"],
    df["Calculated_Sales"],
    rtol=1e-05,
    atol=1e-08
)

print("\nSales validation:")
print(df["Sales_Check"].value_counts())

# 13. OUTLIER ANALYSIS
numeric_columns = [
    "Age",
    "Quantity",
    "Unit_Price",
    "Total_Sales"
]

print("\nOutlier Analysis:")

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(
        column,
        "-> Outliers:",
        len(outliers)
    )

# 14. CHECK REPEATED ORDER IDs
print("\nRepeated Order IDs:")

repeated_orders = df[
    df["Order_ID"].duplicated(keep=False)
].sort_values("Order_ID")

print(repeated_orders[
    ["Order_ID", "Order_Date", "Customer_ID", "Product"]
].head(20))

# 15. FINAL DATA CHECK
print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Missing Values:")
print(df.isnull().sum())

# 16. SAVE CLEANED DATASET

df.to_csv(
    "cleaned_sales_dataset.csv",
    index=False
)

df.to_excel(
    "cleaned_sales_dataset.xlsx",
    index=False
)

print("\nCleaned files created successfully!")

print("\nTask 1 completed.")