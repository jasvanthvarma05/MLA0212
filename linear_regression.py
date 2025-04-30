
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define the data
area = np.array([2600, 3000, 3200, 3600, 4000])
price = np.array([550000, 565000, 610000, 680000, 725000])

# Create a DataFrame with the data
df = pd.DataFrame({'area': area, 'price': price})

# Create a LinearRegression model
reg = LinearRegression()

# Fit the model
reg.fit(df[['area']], df['price'])

# Make a prediction for an area of 3300 square feet
predicted_price = reg.predict([[5000]])

# Print the predicted price
print(f"The predicted price for an area of 3300 square feet is: ${predicted_price[0]:,.2f}")
print(reg.coef_)
print(reg.intercept_)
print("result",135.78767123*5000+180616.43835616432)