#support vector machines

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.express as px

st.title("Support Vector Machine (SVM) Classifier")

# Load dataset
data = pd.DataFrame({
    'tumor_size': [1.2, 2.5, 3.1, 4.0, 4.5, 6.2, 7.1, 1.5, 2.3, 5.8],
    'tumor_texture': [3, 5, 6, 7, 8, 10, 12, 4, 5, 11],
    'tumor_density': [4, 6, 8, 10, 11, 13, 15, 5, 6, 12],
    'label': [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
})

X = data[['tumor_size', 'tumor_texture', 'tumor_density']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
st.write("Model Accuracy:", accuracy)

tumor_size = st.number_input("Tumor Size")
tumor_texture = st.number_input("Tumor Texture")
tumor_density = st.number_input("Tumor Density")

if st.button("Predict"):
    prediction = model.predict([[tumor_size, tumor_texture, tumor_density]])
    if prediction == 1:
        st.warning("Tumor is Malignant")
    else:
        st.success("Benign Tumor")
    st.write("Prediction:", prediction)

# Visualize decision boundary
fig = px.scatter(data, x='tumor_size', y='tumor_texture', color='label', title="Cancer Dataset")
st.plotly_chart(fig)
