# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# --- CHANGE 1: Import RandomForestClassifier ---
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

# 1. Load and Prepare Data
try:
    dataset = pd.read_csv("AutomatedLoan.csv")
except FileNotFoundError:
    print("Error: 'AutomatedLoan.csv' not found. Please place it in the same directory.")
    exit()

# --- Basic Preprocessing ---
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
    dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())

# --- Feature Engineering ---
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['LoanAmountlog'] = np.log(dataset['LoanAmount'])
dataset['TotalIncomelog'] = np.log(dataset['TotalIncome'])

dataset = dataset.drop(columns=['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'])

# 2. Define Features (X) and Target (y)
X = dataset.drop(columns=['Loan_Status']).values
y = dataset['Loan_Status'].values
feature_names = dataset.drop(columns=['Loan_Status']).columns.to_list()


# 3. Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Handle Categorical and Numerical Data Preprocessing
categorical_cols_indices = [0, 1, 2, 3, 4, 6, 7] 
encoders = {}

print("Encoding categorical features...")
for i in categorical_cols_indices:
    le = LabelEncoder()
    X_train[:, i] = le.fit_transform(X_train[:, i])
    X_test[:, i] = le.transform(X_test[:, i])
    encoders[i] = le

le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train)
y_test = le_y.transform(y_test)

print("Scaling all features...")
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 5. Train the Model
# --- CHANGE 2: Use the more powerful RandomForestClassifier ---
print("Training Random Forest model...")
# n_estimators=100 means it will build 100 decision trees
RFClassifier = RandomForestClassifier(n_estimators=100, random_state=0)
RFClassifier.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = RFClassifier.predict(X_test)
accuracy = metrics.accuracy_score(y_pred, y_test)
print(f"The accuracy of the Random Forest is: {accuracy:.2f}")

# 7. Save the Model and Preprocessors for the Front-End
print("\nSaving model and preprocessors to disk...")
# --- CHANGE 3: Save the new RandomForest model ---
joblib.dump(RFClassifier, 'model.pkl')
joblib.dump(ss, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(le_y, 'target_encoder.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
print("✅ Artifacts saved successfully!")