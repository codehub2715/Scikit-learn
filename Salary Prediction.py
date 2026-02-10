#Salary Prediction

from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.DataFrame({
  'experience': [1, 2, 3, 4, 5],
  'salary': [30000, 35000, 40000, 45000, 50000]
})

X = data[['experience']]
y = data['salary']

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([[6]])
print("Prediction Salary for 6 years of experience : ", prediction)

#Output:
#Prediction Salary for 6 years of experience :  [55000.]
