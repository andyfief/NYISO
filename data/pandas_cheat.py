# pandas_cheatsheet.py
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

import pandas as pd

# ========== 1. Load CSV ==========
df = pd.read_csv('your_file.csv')

# ========== 2. Preview Data ==========
print("\n🔎 First 10 rows:")
print(df.head(10))  # Preview first 10 rows (default is 5 -> .head())

print("\n📐 DataFrame Shape (rows, columns):")
print(df.shape)

print("\n🧾 DataFrame Info:")
print(df.info())  # Column types and non-null counts

# ========== 3. Missing Values ==========
print("\n❓ Missing Values per Column:")
print(df.isnull().sum())  # Total nulls per column

print("\n🔎 Percentage of Missing Data:")
print(df.isnull().mean() * 100)

# ========== 4. Basic Statistics ==========
print("\n📊 Basic Descriptive Statistics (numerical columns):")
print(df.describe())

# For all columns including non-numeric:
print("\n📊 Descriptive Statistics (all columns):")
print(df.describe(include='all'))

# ========== 5. Unique Values ==========
print("\n🔢 Unique values per column:")
for col in df.columns:
    unique_vals = df[col].nunique()
    print(f"{col}: {unique_vals} unique values")

# ========== 6. Value Counts ==========
# Get frequency of unique values in a specific column
# Example: Replace 'column_name' with your actual column name
column_name = 'column_name' 
if column_name in df.columns:
    print(f"\n📈 Value Counts for '{column_name}':")
    print(df[column_name].value_counts(dropna=False))

# ========== 7. Mean, Standard Deviation, Median ==========
print("\n📏 Mean of numeric columns:")
print(df.mean(numeric_only=True))

print("\n📐 Standard Deviation:")
print(df.std(numeric_only=True))

print("\n📍 Median:")
print(df.median(numeric_only=True))

# ========== 8. Correlation ==========
print("\n📈 Correlation Matrix:")
print(df.corr(numeric_only=True))

# ========== 9. Column Data Types ==========
print("\n📂 Column Data Types:")
print(df.dtypes)

# ========== 10. Convert Columns ==========
# Convert a column to datetime, if needed
# df['date_column'] = pd.to_datetime(df['date_column'])

# Convert a column to category type (saves memory)
# df['category_column'] = df['category_column'].astype('category')

# ========== 11. Save Cleaned Data ==========
# df.to_csv('cleaned_file.csv', index=False)
