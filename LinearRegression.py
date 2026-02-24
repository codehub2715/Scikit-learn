#Linear Regression

from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.DataFrame({
    'study_hours': [1, 2, 3, 4, 5],
    'score': [40, 50, 60, 70, 80]
})

X = data[['study_hours']]
y =data[['score']]
model = LinearRegression()

model.fit(X,y)
print("Prediction for 6 hours:", model.predict([[6]]))

#Slope And Intercept
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

#Prediction for 6 hours: [[90.]]
#Slope: [[10.]]
#Intercept: [30.]