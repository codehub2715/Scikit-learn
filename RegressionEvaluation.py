#Regression Evaluation

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = [3, 5, 7, 9]
y_pred = [2.5, 5.1, 6.8, 9.2]

print("Mean Squared Error:", mean_squared_error(y_true, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_true, y_pred))
print("R2 Score:", r2_score(y_true, y_pred))
