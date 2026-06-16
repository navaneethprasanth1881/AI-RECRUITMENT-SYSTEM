import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load Dataset
df = pd.read_csv("recruitment_dataset_100_students.csv")

# Encode Target
encoder = LabelEncoder()

df["Selected"] = encoder.fit_transform(df["Selected"])

# Features
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

# Target
y = df["Selected"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ANN Model
model = Sequential()

model.add(Dense(64, activation="relu", input_dim=12))

model.add(Dense(32, activation="relu"))

model.add(Dense(3, activation="softmax"))

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8
)

# Save Model
model.save("ann_selection_model.h5")

# Save Encoder
with open("ann_label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("ANN model saved successfully!")