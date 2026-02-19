import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load dataset
data = pd.read_csv("loan_data.csv")

# Encode text columns
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["Married"] = le.fit_transform(data["Married"])
data["Loan_Status"] = le.fit_transform(data["Loan_Status"])

# Features & Target
X = data[["Gender", "Married", "ApplicantIncome", "LoanAmount", "Credit_History"]]
y = data["Loan_Status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
