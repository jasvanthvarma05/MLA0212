"""3) Perform a classification module on iris dataset and represent the evaluation metrics on its classification using k nearest neighbour. 
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

inf = load_iris()
df = pd.DataFrame(data=inf.data, columns=inf.feature_names)
print(df.columns)
x = inf.data 
y = inf.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
classi = classification_report(y_test,pred)
confu = confusion_matrix(y_test,pred)
print(acc,classi,confu)



# 4) 4) Write a program to demonstrate the working of decision tree using ID3 algorithm. Use an appropriate data for building the decision tree and classify the samples based on it.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

x = df.drop("PlayTennis",axis=1)
y = df["PlayTennis"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state= 42)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
print(acc)



#5) Build a artificial neural network by implementing backpropagation and test the same using appropriate dataset.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize and train MLP (ANN)
model = MLPClassifier(hidden_layer_sizes=(10), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


17)


import numpy as np
from hmmlearn import hmm

# Simulated time series observations (e.g., user activity levels)
observations = np.array([[0], [1], [2], [1], [0], [2], [1], [0]])
model = hmm.MultinomialHMM(n_components=3, n_iter=100)
model.fit(observations)

# Predict hidden states
logprob, hidden_states = model.decode(observations)
print("Hidden States:", hidden_states)

# Transition probabilities
print("Transition Matrix:\n", model.transmat_)
"""

