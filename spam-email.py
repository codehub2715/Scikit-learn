# Spam Email Classifier (Streamlit App)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

st.title("📧 Spam Email Classifier")

st.write("This model predicts whether an email is **Spam** or **Not Spam (Ham)**")

# Sample dataset
data = pd.DataFrame({
    'text': [
        'Congratulations you have won a lottery',
        'You won a free ticket',
        'Please call me when you reach',
        'Let us meet tomorrow',
        'Claim your free prize now',
        'Are you coming to office today?'
    ],
    'label': ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
})

st.subheader("Training Dataset")
st.write(data)

# Encode labels
le = LabelEncoder()
data['label_num'] = le.fit_transform(data['label'])

# Vectorize text
cv = CountVectorizer()
X = cv.fit_transform(data['text'])
y = data['label_num']

# Train model
model = MultinomialNB()
model.fit(X, y)

st.success("Model trained successfully!")


# sidebar in Example Emails
st.sidebar.subheader("Example Emails")
st.sidebar.write("1. Congratulations you have won a lottery")
st.sidebar.write("2. You won a free ticket")
st.sidebar.write("3. Please call me when you reach")
st.sidebar.write("4. Let us meet tomorrow")
st.sidebar.write("5. Claim your free prize now")
st.sidebar.write("6. Are you coming to office today?")

# User Input
st.subheader("Try Your Own Email")

user_email = st.text_input("Enter Email Text")

if st.button("Predict"):

    if user_email.strip() != "":
        email_vec = cv.transform([user_email])
        prediction = model.predict(email_vec)
        result = le.inverse_transform(prediction)[0]

        if result == "spam":
            st.error("🚨 This Email is SPAM")
        else:
            st.success("✅ This Email is NOT SPAM (Ham)")
    else:
        st.warning("Please enter email text.")