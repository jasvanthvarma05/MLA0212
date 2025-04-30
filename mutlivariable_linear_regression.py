import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import plot
area = np.array([2600,3000,3200,4000,5000])
bedroom = np.array([3,4,3,3,4])
age = np.array([20,25,28,30,6])
price = np.array([550000, 565000, 610000, 680000, 725000])
data = {
    'area': area,
    'bedroom': bedroom,
    'age': age,
    'price': price
}
df = pd.DataFrame(data)
print(df)
model = LinearRegression()
model.fit(df[['area','bedroom','age']],df['price'])
pre=model.predict([[2600,4,20]])
print(pre)
print(model.coef_)
print(model.intercept_)