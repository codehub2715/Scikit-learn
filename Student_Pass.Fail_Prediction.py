from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    'study_hours': [2, 5, 10, 1, 8],
    'sleep_hours': [6, 7, 5, 9, 6],
    'pass_exam': [0, 1, 1, 0, 1]
})

X = data[['study_hours', 'sleep_hours']]
y = data['pass_exam']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

print("Predicted", model.predict(X_test))
print("Accuracy:", model.score(X_test, y_test))
