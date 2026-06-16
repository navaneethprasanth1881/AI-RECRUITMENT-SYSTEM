import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("recruitment_dataset_100_students.csv")

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

y = df["PerformanceScore"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

with open("performance_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Performance regression model saved successfully!")