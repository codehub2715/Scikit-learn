#This is your final project for the 15-day Scikit-learn course.

# You will build a complete **House Price Prediction** model using:
# - Data cleaning
# - Preprocessing
# - Feature scaling
# - OneHotEncoding
# - Train/test split
# - Linear Regression
# - Pipeline implementation

# Goal: Predict house prices based on area, bedrooms, location, and age.

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import streamlit as st


def main():
    st.title("House Price Prediction")

    data = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500, 1200, 1800],
    'bedrooms': [2, 3, 3, 4, 2, 3],
    'location': ['city', 'city', 'suburb', 'suburb', 'village', 'village'],
    'age': [5, 10, 8, 15, 20, 12],
    'bathrooms' : [1, 2, 1, 2, 1, 2],
    'parking_spaces' : [1, 2, 1, 2, 1, 2],
    'price': [250000, 300000, 280000, 320000, 220000, 270000]
})
    

    X = data[['area', 'bedrooms', 'location', 'age', 'bathrooms', 'parking_spaces']]
   
    y = data['price']

    numeric_features = ['area', 'bedrooms', 'age', 'bathrooms', 'parking_spaces']
    categorical_features = ['location']

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])


    model.fit(X, y)
    st.success("Model trained successfully!")


    area = st.number_input("Area")
    bedrooms = st.number_input("Bedrooms")
    location = st.selectbox("Location", ['city', 'suburb', 'village'])
    age = st.number_input("Age")
    bathrooms = st.number_input("Bathrooms")
    parking_spaces = st.number_input("Parking Spaces")


    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'location': [location],
        'age': [age],
        'bathrooms': [bathrooms],
        'parking_spaces': [parking_spaces]
    })
    
    if st.button("Predict"):
        prediction = model.predict(input_data)
    
        st.write("Predicted Price:", prediction)

if __name__ == "__main__":
    main()
