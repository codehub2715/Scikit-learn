#Try predicting mobile prices using RAM size:
#- RAM (GB): [2, 4, 6, 8]
#- Price (₹): [8000, 12000, 18000, 24000]
#Build a model and predict price for a 10GB RAM phone.

from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.DataFrame({
    'RAM': [2, 4, 6, 8],
    'Price': [8000, 12000, 18000, 24000]
})

X = data[['RAM']]
y = data['Price']

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([[10]])
print("Prediction For 10GB RAM :", prediction)

#Output:
#Prediction For 10GB RAM : [29000.]