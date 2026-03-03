#We will group customers based on:
#- annual_income
#- spending_score
#
#Goal: Find natural customer groups for business insights.


import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import plotly.express as px

st.title("K-Means Clustering for Customer Segmentation")

data = pd.DataFrame({
    'annual_income': [15, 16, 17, 25, 26, 27, 60, 62, 63, 64],
    'spending_score': [39, 81, 6, 77, 40, 6, 50, 49, 48, 52],
    'age': [22,25,24,32,35,36,45,43,46,47]
})

X = data[['annual_income', 'spending_score', 'age']]
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

data['cluster'] = kmeans.labels_
st.write("Clustered Data:", data)

st.write("Cluster labels:", kmeans.labels_)

fig = px.scatter(data, x='annual_income', y='spending_score', color='cluster', title="Customer Segmentation")
st.plotly_chart(fig)


