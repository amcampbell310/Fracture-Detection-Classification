from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from image_classification import X_test, y_test
# Load the saved model
loaded_model = load_model("")

# Assuming you already have X_test and y_test loaded with your test data

# Make predictions on the test set
y_pred_probs = loaded_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype('int32')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non_fractured', 'Fractured'], yticklabels=['Non_fractured', 'Fractured'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Non_fractured', 'Fractured']))


