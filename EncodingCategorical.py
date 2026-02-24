#Encoding Categorical Data

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

X = pd.DataFrame({
    'city': ['Delhi', 'Mumbai', 'Delhi', 'Chennai']
})

e = OneHotEncoder(sparse_output=False)
encoded = e.fit_transform(X)

print(encoded)
print(e.get_feature_names_out())

#[[0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]]
# ['city_Chennai' 'city_Delhi' 'city_Mumbai']


#label encoding
print("\nLabel Encoding:")
labels = ['spam', 'ham', 'ham', 'spam']
le = LabelEncoder()
y = le.fit_transform(labels)
print(y)

# Label Encoding:
# [1 0 0 1]
