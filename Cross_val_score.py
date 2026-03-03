#Cross_val_score()

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target

model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X, y, cv=5)

print("Cross-Validation Scores:", scores)
print("Average Score:", scores.mean())
