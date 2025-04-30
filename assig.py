import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Step 1: Load the dataset
data = pd.DataFrame({
    'Transaction_id': [12345, 54321, 98765, 24680, 13579],
    'Amount': [100, 50, 200, 75, 150],
    'Merchant_ID': [54321, 98765, 12345, 13579, 24680],
    'Transaction_time': [12, 3, 5, 7, 9],
    'Fraudulent': ['no', 'yes', 'no', 'no', 'yes']
})

# Step 2: Encode categorical variables
label_encoder = LabelEncoder()
data['Fraudulent'] = label_encoder.fit_transform(data['Fraudulent'])

# Step 3: Define features and target variable
features = ['Amount', 'Merchant_ID', 'Transaction_time']
X = data[features]
y = data['Fraudulent']

# Step 4: Train the logistic regression model on the entire dataset
model = LogisticRegression()
model.fit(X, y)

# Step 5: Make predictions on the entire dataset
data['Predicted_Fraudulent'] = model.predict(X)

# Step 6: Add predicted probabilities for fraudulent transactions
data['Fraud_Probability'] = model.predict_proba(X)[:, 1]

# Step 7: Display the dataset with predictions
print(data)
