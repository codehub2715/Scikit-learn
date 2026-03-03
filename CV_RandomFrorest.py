#Cross Validation with Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()

scores = cross_val_score(model, X, y, cv=10)
print("Scores:", scores)
print("Mean Accuracy:", scores.mean())
