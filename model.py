import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("loan_data.csv")

# Encode categorical columns
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Married'] = le.fit_transform(data['Married'])
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])

# Features and Target
X = data[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = data['Loan_Status']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")
