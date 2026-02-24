#Data Preprocessing: Handling Missing Values

import numpy as np
from sklearn.impute import SimpleImputer

X = [[1, 2], [3, np.nan], [7, 6]]

imputer = SimpleImputer(strategy='mean')
X_transformed = imputer.fit_transform(X)

print(X_transformed)
