import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("titanic.csv")

# Show first rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# -----------------------------
# DATA CLEANING
# -----------------------------

# Fill missing age values
df['age'].fillna(df['age'].mean(), inplace=True)

# Drop cabin column (if exists)
if 'cabin' in df.columns:
    df.drop('cabin', axis=1, inplace=True)

# -----------------------------
# DATA ANALYSIS (GRAPHS)
# -----------------------------

# 1. Survival count
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

# 2. Survival vs Gender
sns.countplot(x='survived', hue='sex', data=df)
plt.title("Survival by Gender")
plt.show()

# 3. Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()