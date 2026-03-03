#Titanic Survival Prediction

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


st.title("Titanic Survival Prediction")

data = pd.DataFrame({
    'age': [22, 38, 26, 35, 28, 42],
    'fare': [7.25, 71.28, 8.05, 53.1, 8.46, 26.55],
    'gender': ['male', 'female', 'female', 'female', 'male', 'male'],
    'class': ['3rd', '1st', '3rd', '1st', '3rd', '2nd'],
    'survived': [0, 1, 1, 1, 0, 0]
})

X = data[['age', 'fare', 'gender', 'class']]
y = data['survived']

numeric_features = ['age', 'fare']
catagorical_features = ['gender', 'class']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

catagorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', catagorical_transformer, catagorical_features)
])

#pipeine
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

model.fit(X,y)

age = st.number_input("Age")
fare = st.number_input("Fare")
gender = st.selectbox("Gender", ['male', 'female'])
Class = st.selectbox("Class", ['1st', '2nd', '3rd'])
if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'fare': [fare],
        'gender': [gender],
        'class': [Class]
    })

    prediction = model.predict(input_data)

    st.write("Prediction:", prediction[0])

    if prediction[0] == 1:
        st.success("Survived")
    else:
        st.error("Died")

param_grid = {
    'classifier__C': [0.1, 1, 10]
}

grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X, y)
st.write("Best Params:", grid.best_params_)