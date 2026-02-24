#Feature Scaling : Standardization

from sklearn.preprocessing import StandardScaler , MinMaxScaler

X = [[100], [200], [300]]

scaler = StandardScaler()
scaled = scaler.fit_transform(X)
print("Standardized Data:\n", scaled)

#Standardized Data:
#  [[-1.22474487]
#  [ 0.        ]
#  [ 1.22474487]]

#Min-Max Scaling

Y = [[10], [20], [30]]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(Y)
print("\nMin-Max Scaled Data:\n", scaled)

# Min-Max Scaled Data:
#  [[0. ]
#  [0.5]
#  [1. ]]