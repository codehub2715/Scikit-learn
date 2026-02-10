#Predict the fruit calories for a given csv file

from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv('fruit.csv')
X = data[['fruit_id']]
y = data['calories']

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([[6]])
print("Prediction for 6th fruit : ", prediction)

#Output:
#Prediction for 6th fruit :  [600.]