#Sales Prediction
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'ads_budget': [1000, 1500, 2000, 2500, 3000],
    'sales': [5000, 7000, 9000, 11000, 13000]
})

X = data[['ads_budget']]
y= data['sales']

model = LinearRegression()
model.fit(X,y)

print("Predicted Sales for 3500 budget: ", model.predict([[3500]]))

#Predicted Sales for 3500 budget:  [15000.]