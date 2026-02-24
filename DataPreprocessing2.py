#**Tasks:**
# 1. Fill missing values.
# 2. Encode gender using OneHotEncoder.
# 3. Scale the income column.

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.DataFrame({
    'age': [20, None, 30],
    'gender': ['M', 'F', 'M'],
    'income': [40000, 50000, None]
})

numeric_features = ['age', 'income']
categorical_features = ['gender']

numeric_transformer = Pipeline([
    ('imputer',SimpleImputer(strategy = 'mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

processed = preprocessor.fit_transform(data)
print(processed)
print(preprocessor.get_feature_names_out())

#Output:
#[[-1.22474487 -1.22474487  0.          1.        ]
# [ 0.          1.22474487  1.          0.        ]
# [ 1.22474487  0.          0.          1.        ]]
#['num__age' 'num__income' 'cat__gender_F' 'cat__gender_M']