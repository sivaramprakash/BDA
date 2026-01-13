import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# 1. Load the dataset
# If you are using a file, use: data = pd.read_csv("weather.csv")
data = pd.read_csv("../dataset/weather.csv")

# 2. Preprocessing
df = data.copy()
df['play'] = df['play'].map({'yes': 1, 'no': 0})
df['humidity'] = df['humidity'].map({'high': 1, 'normal': 0})
df['outlook'] = df['outlook'].map({'sunny': 0, 'overcast': 1, 'rainy': 2})
df['temperature'] = df['temperature'].map({'hot': 0, 'mild': 1, 'cool': 2})
df['windy'] = df['windy'].apply(lambda x: 1 if str(x).upper() == "TRUE" else 0)

# 3. Split Features and Target
X = df.drop("play", axis=1)
y = df['play']

# 4. Train Decision Tree Model
dt_model = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_model.fit(X, y)

# --- VISUALIZATIONS ---

# Visualization 1: Class Distribution (Seaborn)
plt.figure(figsize=(6, 4))
sns.countplot(x='play', data=data) # Using original data for labels 'yes'/'no'
plt.title('Distribution of Play (Target Variable)')
plt.savefig('class_distribution.png')
plt.show()

# Visualization 2: The Decision Tree (Scikit-Learn)
plt.figure(figsize=(15, 10))
plot_tree(dt_model, 
          feature_names=X.columns.tolist(), 
          class_names=['No (0)', 'Yes (1)'], 
          filled=True, 
          rounded=True, 
          fontsize=12)
plt.title('Decision Tree structure for Weather Dataset')
plt.savefig('decision_tree_plot.png')
plt.show()