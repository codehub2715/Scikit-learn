#Clean + Encode + Scale

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.DataFrame({
    'age': [20, 25, None, 30],
    'salary': [30000, 50000, 45000, None],
    'city': ['Delhi', 'Mumbai', 'Chennai', 'Delhi']
})

numeric_features = ['age', 'salary']
categorical_features = ['city']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ('num' , numeric_transformer, numeric_features),
    ('cat' , categorical_transformer, categorical_features)
])

processed = preprocessor.fit_transform(data)
print(processed)
print(preprocessor.get_feature_names_out())

#[[-1.41421356 -1.58518785  0.          1.          0.        ]
# [ 0.          1.13227703  0.          0.          1.        ]
# [ 0.          0.45291081  1.          0.          0.        ]
# [ 1.41421356  0.          0.          1.          0.        ]]
#['num__age' 'num__salary' 'cat__city_Chennai' 'cat__city_Delhi'
# 'cat__city_Mumbai']