# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier with criterion 'entropy' to use the ID3 algorithm
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Print the textual representation of the tree
tree_rules = export_text(clf, feature_names=iris.feature_names)
print(tree_rules)

# Evaluate the classifier
accuracy = clf.score(X_test, y_test)
print(f"Accuracy of the Decision Tree classifier: {accuracy * 100:.2f}%")

# Classify new samples
new_samples = np.array([[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [7.2, 3.6, 6.1, 2.5]])
predictions = clf.predict(new_samples)
predicted_classes = [iris.target_names[p] for p in predictions]

print(f"New samples predictions: {predicted_classes}")
