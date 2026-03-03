import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


conn = sqlite3.connect("house_data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    area REAL,
    bedrooms INTEGER,
    bathrooms INTEGER,
    parking INTEGER,
    location TEXT,
    age INTEGER,
    predicted_price REAL
)
""")
conn.commit()


np.random.seed(42)
# Generate random data
data = pd.DataFrame({
    'area': np.random.randint(800, 3000, 200),
    'bedrooms': np.random.randint(1, 6, 200),
    'bathrooms': np.random.randint(1, 4, 200),
    'parking': np.random.randint(0, 3, 200),
    'location': np.random.choice(['city', 'suburb', 'village'], 200),
    'age': np.random.randint(1, 30, 200),
})

# Price formula 
data['price'] = (
    data['area'] * 50 +
    data['bedrooms'] * 50000 +
    data['bathrooms'] * 30000 +
    data['parking'] * 20000 -
    data['age'] * 1000 +
    np.random.randint(-20000, 20000, 200)
)

# Split
X = data.drop("price", axis=1)
y = data["price"]

# Preprocessing
numeric_features = ['area', 'bedrooms', 'bathrooms', 'parking', 'age']
categorical_features = ['location']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

#pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 Internship-Level House Price Prediction System")

st.success("Model trained successfully!")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", 500, 5000, 1500)
    bedrooms = st.number_input("Bedrooms", 1, 6, 3)
    bathrooms = st.number_input("Bathrooms", 1, 4, 2)

with col2:
    parking = st.number_input("Parking Spaces", 0, 3, 1)
    location = st.selectbox("Location", ['city', 'suburb', 'village'])
    age = st.number_input("House Age (years)", 0, 50, 5)

st.markdown("---")

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'parking': [parking],
        'location': [location],
        'age': [age]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    st.subheader(f"💰 Predicted Price: ₹ {prediction:,.0f}")

    # Save to database
    cursor.execute("""
    INSERT INTO predictions (area, bedrooms, bathrooms, parking, location, age, predicted_price)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (area, bedrooms, bathrooms, parking, location, age, prediction))
    conn.commit()

    st.success("Prediction saved to database!")

st.markdown("---")

st.subheader("📊 Model Performance")
st.write(f"MAE: ₹ {mae:,.0f}")
st.write(f"R² Score: {r2:.2f}")

st.markdown("---")

if st.checkbox("Show Prediction History"):
    df_history = pd.read_sql_query("SELECT * FROM predictions", conn)
    st.dataframe(df_history)