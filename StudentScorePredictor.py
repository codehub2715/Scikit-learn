# **Student Score Predictor**.

# Goal: Predict a student's exam score based on:
# - Study hours
# - Sleep hours
# - Number of practice tests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Load Data 
data = pd.DataFrame({
    'study_hours': [2, 3, 4, 5, 6, 7],
    'sleep_hours': [7, 6, 8, 5, 7, 6],
    'practice_tests': [1, 2, 2, 3, 3, 4],
    'internet_hours': [1, 2, 1, 3, 2, 4],
    'score': [50, 55, 65, 70, 78, 85]
})

print(data)

#Step 2: Split Features & Labels
X = data[['study_hours', 'sleep_hours', 'practice_tests']]
y = data['score']

#Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#Step 4: Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

#Step 5: Make Predictions

prediction = model.predict(X_test)
print("Predicted Scores:", prediction)

print("Custom Prediction:", model.predict([[7, 7, 4]]))

#Step 6: Evaluate The Model
score = model.score(X_test, y_test)
print("R² Score:", score)

#Visualization 

plt.scatter(data['study_hours'], data['score'])
plt.plot(data['study_hours'], model.predict(data[['study_hours', 'sleep_hours', 'practice_tests']]), linestyle='dashed')
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.title("Study Hours vs Score Prediction")
plt.show()

#   study_hours  sleep_hours  practice_tests  internet_hours  score
#0            2            7               1               1     50
#1            3            6               2               2     55
#2            4            8               2               1     65
#3            5            5               3               3     70
#4            6            7               3               2     78
#5            7            6               4               4     85
#Training samples: 4
#Testing samples: 2
#Model trained successfully!
#Predicted Scores: [66.25 67.75]
#Custom Prediction: [86.75]
#R² Score: 0.47