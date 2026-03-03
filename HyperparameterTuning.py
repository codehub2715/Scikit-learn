#Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import streamlit as st

st.title("Hyperparameter Tuning")

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()

#GridSearchCV
st.subheader("GridSearchCV")
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

st.write("GridSearch Best Parameters:", grid_search.best_params_)
st.write("GridSearch Best Score:", grid_search.best_score_)

#RandomizedSearchCV
st.subheader("RandomizedSearchCV")
param_dist = {
    'n_estimators': np.arange(10, 200),
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

random = RandomizedSearchCV(model, param_distributions=param_dist, cv=5,random_state=42)
random.fit(X, y)

st.write("RandomizedSearch Best Parameters:", random.best_params_)
st.write("RandomizedSearch Best Score:", random.best_score_)
