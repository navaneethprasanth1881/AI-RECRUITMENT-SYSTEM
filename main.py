import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("recruitment_dataset_100_students.csv")

# Convert Selected column into numbers
encoder = LabelEncoder()
df["Selected"] = encoder.fit_transform(df["Selected"])

# Input features
X = df[[
    "Python",
    "SQL",
    "MachineLearning",
    "Communication",
    "CGPA",
    "Experience_Years",
    "Projects_Count",
    "Certifications",
    "Internships",
    "AptitudeScore",
    "CodingScore",
    "InterviewScore"
]]

# Target output
y = df["Selected"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Report
print(classification_report(y_test, y_pred))




import pickle

# Save trained model
with open("selection_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save label encoder
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("Model saved successfully!")