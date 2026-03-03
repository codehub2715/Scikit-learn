#Evaluate Logistic Regression

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import streamlit as st

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)

st.write("Accuracy:", accuracy)
st.write("Confusion Matrix:\n", confusion)
st.write("Classification Report:\n", classification)

#Cross Validation
scores = cross_val_score(model,X,y,cv=5)

st.write("Cross-Validation Scores:", scores)
st.write("Average Score:", scores.mean())