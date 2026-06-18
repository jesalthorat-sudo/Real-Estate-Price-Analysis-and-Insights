import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data.csv")

# Basic Information
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

# Clean Column Names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_", regex=False)
)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert Numeric Columns
df["price"] = pd.to_numeric(
    df["price"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
).round(0).astype("Int64")

df["area"] = pd.to_numeric(
    df["area"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
).astype("Int64")

df["rate_per_sqft"] = pd.to_numeric(
    df["rate_per_sqft"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
).astype("Int64")

# Clean Text Columns
df["status"] = df["status"].str.strip().str.lower()
df["rera_approval"] = df["rera_approval"].str.strip().str.lower()
df["flat_type"] = df["flat_type"].str.strip().str.lower()

# Remove Duplicates
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:", df.shape)

# Descriptive Statistics
print("\nSummary Statistics:")
print(df.describe())

# Most Expensive Property
print("\nMost Expensive Property:")
print(df.loc[df["price"].idxmax()])

# Average Price by Locality
print("\nAverage Price by Locality:")
print(
    df.groupby("locality")["price"]
      .mean()
      .sort_values(ascending=False)
)

# Average Rate per Sqft by Locality
print("\nAverage Rate per Sqft by Locality:")
print(
    df.groupby("locality")["rate_per_sqft"]
      .mean()
      .sort_values(ascending=False)
)

# Median Price by Status
print("\nMedian Price by Status:")
print(df.groupby("status")["price"].median())

# Median Price by RERA Approval
print("\nMedian Price by RERA Approval:")
print(df.groupby("rera_approval")["price"].median())

# Average Price by BHK
print("\nAverage Price by BHK:")
print(df.groupby("bhk_count")["price"].mean())

# Average Price by Flat Type
print("\nAverage Price by Flat Type:")
print(df.groupby("flat_type")["price"].mean())

# Average Price by Builder
print("\nAverage Price by Company:")
print(
    df.groupby("company_name")["price"]
      .mean()
      .sort_values(ascending=False)
)

# Correlation Matrix
plt.figure(figsize=(8, 5))
numeric_df = df.select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot: Area vs Price
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="area",
    y="price",
    hue="bhk_count"
)

plt.title("Area vs Price")
plt.show()

# Scatter Plot: Area vs Rate Per Sqft
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="area",
    y="rate_per_sqft"
)

plt.title("Area vs Rate Per Sqft")
plt.show()

# Box Plot for Price
plt.figure(figsize=(8, 4))

sns.boxplot(x=df["price"])

plt.title("Price Distribution")
plt.show()

# Top 10 Expensive Localities
top_localities = (
    df.groupby("locality")["price"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10, 5))

top_localities.plot(kind="bar")

plt.title("Top 10 Localities by Average Price")
plt.ylabel("Average Price")
plt.xticks(rotation=45)

plt.show()