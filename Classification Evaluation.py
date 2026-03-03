from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred))

# Output:
Accuracy: 0.6
# Confusion Matrix:
#  [[1 1]
#  [1 2]]
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.50      0.50      0.50         2
#            1       0.67      0.67      0.67         3

#     accuracy                           0.60         5
#    macro avg       0.58      0.58      0.58         5
# weighted avg       0.60      0.60      0.60         5

