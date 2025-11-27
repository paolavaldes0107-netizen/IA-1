import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import time

# Load training data
train_csv_path = r"IA\datasets\mnist\mnist_train.csv"
mnist_train_df = pd.read_csv(train_csv_path)

# Load test data
test_csv_path = r"IA\datasets\mnist\mnist_test.csv"
mnist_test_df = pd.read_csv(test_csv_path)

# Split features and labels
X_train = mnist_train_df.iloc[:, 1:]
y_train = mnist_train_df.iloc[:, 0]
X_test = mnist_test_df.iloc[:, 1:]
y_test = mnist_test_df.iloc[:, 0]

# Create and train the SVM classifier
clf = svm.SVC()

# Measure training time
start_train_time = time.time()
clf.fit(X_train, y_train)
end_train_time = time.time()
train_time = end_train_time - start_train_time
print("Training time (seconds):", train_time)

# Predict on the test set
start_pred_time = time.time()
y_pred = clf.predict(X_test)
end_pred_time = time.time()
pred_time = end_pred_time - start_pred_time
print("Prediction time (seconds):", pred_time)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)