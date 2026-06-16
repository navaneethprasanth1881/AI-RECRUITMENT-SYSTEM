import streamlit as st
import pickle
import numpy as np
import pdfplumber
import re
import pandas as pd
import os
import ollama
from tensorflow.keras.models import load_model
# Load ML classification model
with open("selection_model.pkl", "rb") as file:
    selection_model = pickle.load(file)

# Load label encoder
with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# Load regression model
with open("performance_model.pkl", "rb") as file:
    performance_model = pickle.load(file)
# Load ANN model

ann_model = load_model("ann_selection_model.h5")

# Load ANN encoder
with open("ann_label_encoder.pkl", "rb") as file:
    ann_encoder = pickle.load(file)
st.title("AI Recruitment & Talent Analytics Platform")
st.subheader("Upload Candidate Resume")
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
python_skill = 0
sql_skill = 0
ml_skill = 0
candidate_name = "Not Found"
candidate_email = "Not Found"
candidate_phone = "Not Found"
if uploaded_file is not None:
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    text_lower = text.lower()

    if "python" in text_lower:
        python_skill = 1

    if "sql" in text_lower:
        sql_skill = 1

    if "machine learning" in text_lower or "ml" in text_lower:
        ml_skill = 1

    email_match = re.search(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        text
    )

    if email_match:
        candidate_email = email_match.group()

    phone_pattern = r"(?:\+91[\s\-]?)?(?:\d[\s\-]?){10,12}"
    phone_matches = re.findall(phone_pattern, text)

    clean_numbers = []
    for number in phone_matches:
        digits = re.sub(r"\D", "", number)

        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]

        if len(digits) == 10 and digits[0] in "6789":
            clean_numbers.append(digits)

    clean_numbers = list(set(clean_numbers))

    if clean_numbers:
        candidate_phone = ", ".join(clean_numbers)

    lines = text.split("\n")

    for line in lines:
        clean_line = line.strip()

        if (
            clean_line
            and "@" not in clean_line
            and not re.search(r"\d", clean_line)
            and len(clean_line.split()) <= 4
        ):
            candidate_name = clean_line
            break

    st.success("Resume processed successfully!")

    st.subheader("Candidate Profile")
    st.write("Name:", candidate_name)
    st.write("Email:", candidate_email)
    st.write("Phone:", candidate_phone)

    st.subheader("Detected Skills")
    st.write("Python:", python_skill)
    st.write("SQL:", sql_skill)
    st.write("Machine Learning:", ml_skill)

    st.subheader("Extracted Resume Text")
    st.text(text[:1500])

st.subheader("Enter Candidate Details")

communication = st.slider("Communication Score", 1, 10)
cgpa = st.number_input("CGPA", 0.0, 10.0)
experience = st.number_input("Experience Years", 0, 10)
projects = st.number_input("Projects Count", 0, 20)
certifications = st.number_input("Certifications", 0, 20)
internships = st.number_input("Internships", 0, 10)
aptitude = st.number_input("Aptitude Score", 0, 100)
coding = st.number_input("Coding Score", 0, 100)
interview = st.number_input("Interview Score", 0, 100)

if st.button("Predict Candidate"):

    data = np.array([[
        python_skill,
        sql_skill,
        ml_skill,
        communication,
        cgpa,
        experience,
        projects,
        certifications,
        internships,
        aptitude,
        coding,
        interview
    ]])

    selection_prediction = selection_model.predict(data)
    selection_result = encoder.inverse_transform(selection_prediction)[0]

    performance_prediction = performance_model.predict(data)
    performance_score = round(float(performance_prediction[0]), 2)
    ann_prediction = ann_model.predict(data, verbose=0)
    ann_class = np.argmax(ann_prediction)
    ann_result = ann_encoder.inverse_transform([ann_class])[0]

    if selection_result == "Yes" and performance_score >= 80:
        candidate_rank = "High Potential"

    elif selection_result == "Likely Selected" or performance_score >= 60:
        candidate_rank = "Medium Potential"

    else:
        candidate_rank = "Low Potential"

    st.success(f"ML Selection Prediction: {selection_result}")
    st.info(f"Expected Performance Score: {performance_score}")
    st.warning(f"ANN Deep Learning Prediction: {ann_result}")
    st.success(f"Candidate Rank: {candidate_rank}")

    skills = []

    if python_skill == 1:
        skills.append("Python")

    if sql_skill == 1:
        skills.append("SQL")

    if ml_skill == 1:
        skills.append("Machine Learning")
    

    if len(skills) == 0:
        skill_text = "general technical skills"
    else:
        skill_text = ", ".join(skills)

    prompt = f"""
            Generate exactly 5 short technical interview questions for a candidate skilled in {skill_text}.

            Give only numbered questions."""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        st.subheader("AI Generated Interview Questions")
        st.write(response["message"]["content"])  
    except Exception:
        st.error(
            "Ollama is not running. Run this in terminal: ollama run llama3"
        )

    candidate_data = {
        "Name": [candidate_name],
        "Email": [candidate_email],
        "Phone": [candidate_phone],
        "Python": [python_skill],
        "SQL": [sql_skill],
        "MachineLearning": [ml_skill],
        "CGPA": [cgpa],
        "Experience": [experience],
        "Projects": [projects],
        "Certifications": [certifications],
        "Internships": [internships],
        "AptitudeScore": [aptitude],
        "CodingScore": [coding],
        "InterviewScore": [interview],
        "MLSelectionPrediction": [selection_result],
        "PerformanceScore": [performance_score],
        "ANNPrediction": [ann_result],
        "CandidateRank": [candidate_rank]
    }

    df = pd.DataFrame(candidate_data)

    file_name = "candidate_results.csv"

    if os.path.exists(file_name):
        df.to_csv(
            file_name,
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            file_name,
            index=False
        )

    st.success("Candidate result saved successfully!")
 