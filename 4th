import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Step 1: Load the Dataset
# Create a synthetic dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Encode categorical variables
for column in df.columns:
    df[column] = df[column].astype('category').cat.codes

# Features and target variable
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Implement the ID3 Algorithm
def entropy(y):
    freq = Counter(y)
    total = len(y)
    return -sum((count / total) * np.log2(count / total) for count in freq.values())

def information_gain(X, y, feature):
    original_entropy = entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * entropy(y[X[feature] == values[i]]) for i in range(len(values)))
    return original_entropy - weighted_entropy

def id3(X, y, features):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if len(features) == 0:
        return Counter(y).most_common(1)[0][0]

    best_feature = max(features, key=lambda feature: information_gain(X, y, feature))
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in np.unique(X[best_feature]):
        subtree = id3(X[X[best_feature] == value], y[X[best_feature] == value], remaining_features)
        tree[best_feature][value] = subtree

    return tree

# Step 4: Train the Model
features = list(X_train.columns)
tree = id3(X_train, y_train, features)

def predict(sample, tree):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = sample[feature]
    return predict(sample, tree[feature][value])

# Step 5: Make Predictions
y_pred = X_test.apply(lambda x: predict(x, tree), axis=1)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")
