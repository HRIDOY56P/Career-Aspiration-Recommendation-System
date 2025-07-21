# train_model.py
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

# Load the dataset (update this path if needed)
df = pd.read_csv("student-scores-1.csv")

# Feature engineering
df["total_score"] = df[[
    "math_score", "history_score", "physics_score", 
    "chemistry_score", "biology_score", "english_score"]].sum(axis=1)
df["average_score"] = df["total_score"] / 6

# Drop unnecessary columns
df.drop(columns=['id', 'first_name', 'last_name', 'email'], inplace=True)

# Encode categorical data
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['extracurricular_activities'] = df['extracurricular_activities'].map({False: 0, True: 1})
df['career_aspiration'] = df['career_aspiration'].map({
    'Lawyer': 0, 'Doctor': 1, 'Civil': 2, 'Biotechnology': 3, 'Software Engineering': 4, 'biomedical':5 })

# Prepare features and labels
X = df.drop('career_aspiration', axis=1)
y = df['career_aspiration']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Classifier": SVC(probability=True),
    "Random Forest Classifier": RandomForestClassifier(),
    "K Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier()
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("=" * 50)
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name

print("=" * 50)
print(f"✅ Best Model: {best_model_name} with Accuracy: {best_score:.4f}")

# Save best model and scaler
os.makedirs("code", exist_ok=True)
pickle.dump(best_model, open("code/model.pkl", 'wb'))
pickle.dump(scaler, open("code/scaler.pkl", 'wb'))
print("✅ Saved best model and scaler to 'code/' folder.")
