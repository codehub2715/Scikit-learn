from sklearn.linear_model import LogisticRegression

X = [[10], [15], [30], [35]]  # temperature

y = [0, 0, 1, 1]  # cold = 0, hot = 1

model = LogisticRegression()
model.fit(X, y)

print("Predicted:", model.predict([[25]]))

#Output:
#Predicted: [1] #Hot

