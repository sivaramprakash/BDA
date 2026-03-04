# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../dataset/iris.csv")

# Display first few rows
print(df.head())

# 1. Line Plot - Sepal Length variation
plt.figure()
plt.plot(df['sepal_length'])
plt.title("Line Plot of Sepal Length")
plt.xlabel("Observations")
plt.ylabel("Sepal Length")
plt.show()

# 2. Bar Chart - Average sepal length by species
plt.figure()
df.groupby('species')['sepal_length'].mean().plot(kind='bar')
plt.title("Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length")
plt.show()

# 3. Histogram - Petal Length distribution
plt.figure()
plt.hist(df['petal_length'], bins=10)
plt.title("Histogram of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure()
plt.scatter(df['sepal_length'], df['petal_length'])
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# 5. Box Plot - Sepal Width by Species
plt.figure()
sns.boxplot(x='species', y='sepal_width', data=df)
plt.title("Sepal Width Distribution by Species")
plt.show()
