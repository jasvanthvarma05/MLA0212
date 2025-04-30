import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
age = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
       55, 60, 65, 70, 75, 80, 85, 90, 95])
price = np.array([200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000,
         700000, 750000, 800000, 850000, 900000, 950000, 1000000, 1050000, 1100000])
data = {'age': age,
    'price': price}
df = pd.DataFrame(data)
print(data)
x = df[['age']]
y=df['price']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
print(model.predict(x_test))
print(y_test)
print(model.score(x_test,y_test))