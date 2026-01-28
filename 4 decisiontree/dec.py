

import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("../dataset/weather.csv")

# -----------------------------
# Function to calculate entropy
# -----------------------------
def entropy(column):
    values = column.value_counts()
    total = len(column)
    ent = 0
    for count in values:
        p = count / total
        ent -= p * math.log2(p)
    return ent

# -----------------------------------
# Function to calculate information gain
# -----------------------------------
def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].value_counts().index
    weighted_entropy = 0
    for v in values:
        subset = data[data[feature] == v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy

# (a) Compute Entropy of PlayTennis
target_entropy = entropy(data['play'])
print("(a) Entropy(PlayTennis) =", round(target_entropy, 4))

# (b) Compute Information Gain
features = ['outlook', 'temperature', 'humidity', 'windy']
gain_values = {}

print("\n(b) Information Gain values:")
for f in features:
    g = info_gain(data, f, 'play')
    gain_values[f] = g
    print(f"Gain({f}) =", round(g, 4))

# (c) Identify Root Node
root = max(gain_values, key=gain_values.get)
print("\n(c) Root Node of Decision Tree:", root)

# ------------------------------------------------
# (d) INBUILT Decision Tree (ID3) + TREE IMAGE
# ------------------------------------------------

# Encode categorical features
le = LabelEncoder()
encoded_data = data.copy()
for col in encoded_data.columns:
    encoded_data[col] = le.fit_transform(encoded_data[col])

X = encoded_data[features]
y = encoded_data['play']

# Train decision tree using entropy (ID3)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# Plot decision tree
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree using ID3 (Entropy)")
plt.show()
