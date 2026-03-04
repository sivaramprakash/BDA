import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("../dataset/iris.csv")

# --- DATA CLEANING (Fixes the Errors) ---
# Convert numeric columns that might have '??' or strings to actual numbers
# errors='coerce' turns non-numeric values into NaN (Not a Number)
numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows where Species is missing (NaN) so grouping works correctly
df = df.dropna(subset=['Species'])
# ---------------------------------------

# Display cleaned data info
print(df.head())
print(df.dtypes)

# 1. Line Plot - Sepal Length variation
plt.figure(figsize=(8, 4))
plt.plot(df['SepalLengthCm'])
plt.title("Line Plot of Sepal Length")
plt.xlabel("Observations")
plt.ylabel("Sepal Length")
plt.show()

# 2. Bar Chart - Average sepal length by species
plt.figure(figsize=(8, 4))
# Use numeric_only=True on the whole group to avoid TypeErrors
df.groupby('Species').mean(numeric_only=True)['SepalLengthCm'].plot(kind='bar', color='skyblue')
plt.title("Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length")
plt.xticks(rotation=0)
plt.show()

# 3. Histogram - Petal Length distribution
plt.figure(figsize=(8, 4))
# dropna() ensures the histogram doesn't break on missing values
plt.hist(df['PetalLengthCm'].dropna(), bins=15, color='green', edgecolor='black')
plt.title("Histogram of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(8, 4))
plt.scatter(df['SepalLengthCm'], df['PetalLengthCm'], alpha=0.5)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# 5. Box Plot - Sepal Width by Species
plt.figure(figsize=(8, 4))
# Note: Ensure 'Species' is capitalized to match your CSV
sns.boxplot(x='Species', y='SepalWidthCm', data=df)
plt.title("Sepal Width Distribution by Species")
plt.show()
