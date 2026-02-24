#Create a model to predict whether a mobile phone is expensive (1) or budget (0).
#Predict category for 5GB RAM.

from sklearn.linear_model import LogisticRegression

X = [[2], [4], [6], [8]] #RAM = GB
y = [0, 0, 1, 1]    #expensive (1) or budget (0)

model = LogisticRegression()
model.fit(X, y)

print("Predicted:", model.predict([[5]]))
