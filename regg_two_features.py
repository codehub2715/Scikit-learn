#Predict house price using size & number of rooms.

from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    'size': [1000, 1500, 1800, 2000, 2400],
    'rooms': [2, 3, 4, 4, 5],
    'price': [100000, 150000, 180000, 200000, 250000]
})

X = data[['size', 'rooms']]
y = data['price']

X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)
print("Predicted Price:",model.predict([[2100,4]]))

score = model.score(X_test,y_test)
print("R² Score:", score)

#Predicted Price: [215000.]
#R² Score: nan
