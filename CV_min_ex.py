# 1. Train DecisionTreeClassifier on Iris dataset.
# 2. Evaluate using:
#    - accuracy_score
#    - classification_report
#    - cross_val_score (cv=7)
# 3. Compare with LogisticRegression.

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import streamlit as st

st.title("Model Evaluation")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=0)

st.subheader("Decision Tree Classifier")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred)

scores = cross_val_score(model, X, y, cv=7)

st.write("Accuracy:", accuracy)
st.write("Classification Report:\n", classification)
st.write("Cross-Validation Scores:", scores)
st.write("Average Score:", scores.mean())

#compare with LogisticRegression

from sklearn.linear_model import LogisticRegression

st.subheader("Logistic Regression")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred)
classification_log = classification_report(y_test, y_pred)

scores_log = cross_val_score(model, X, y, cv=7)

st.write("Accuracy:", accuracy_log)
st.write("Classification Report:\n", classification_log)
st.write("Cross-Validation Scores:", scores_log)
st.write("Average Score:", scores_log.mean())