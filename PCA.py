#We will reduce a dataset with 4 features into 2 principal components for easier visualization.

import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px

st.title("Principal Component Analysis (PCA)")

X = pd.DataFrame({
    'height': [150,160,170,180,190],
    'weight': [50,60,70,80,90],
    'age': [18,22,25,30,35],
    'income': [30,40,50,60,70],
    'hours_sleep': [6,7,8,9,10],
    'daily_steps' : [1000,2000,3000,4000,5000]
})

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X)

# Explained Variance
st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
st.write("PCA DataFrame:", pca_df)

fig = px.scatter_3d(pca_result,x= 0, y=1, z=2, color=[0,1,2,3,4])

st.plotly_chart(fig)
