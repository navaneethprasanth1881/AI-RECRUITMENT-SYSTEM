import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("Recruitment Analytics Dashboard")

file_name = "candidate_results.csv"

if not os.path.exists(file_name):
    st.error("No candidate results found. First run app.py and predict candidates.")
    st.stop()

df = pd.read_csv(file_name)

st.subheader("Candidate Database")
st.dataframe(df)
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Candidates", len(df))
col2.metric("High Potential", (df["CandidateRank"] == "High Potential").sum())
col3.metric("Medium Potential", (df["CandidateRank"] == "Medium Potential").sum())

st.subheader("Candidate Ranking Count")

rank_count = df["CandidateRank"].value_counts()

fig, ax = plt.subplots()
ax.bar(rank_count.index, rank_count.values)
ax.set_xlabel("Candidate Rank")
ax.set_ylabel("Number of Candidates")
ax.set_title("Candidate Ranking Distribution")

st.pyplot(fig)

st.subheader("ML Selection Prediction Count")

selection_count = df["MLSelectionPrediction"].value_counts()

fig2, ax2 = plt.subplots()
ax2.bar(selection_count.index, selection_count.values)
ax2.set_xlabel("Prediction")
ax2.set_ylabel("Number of Candidates")
ax2.set_title("ML Selection Prediction Count")
st.pyplot(fig2)

st.subheader("Average Scores")

score_columns = ["AptitudeScore", "CodingScore", "InterviewScore", "PerformanceScore"]
avg_scores = df[score_columns].mean()

fig3, ax3 = plt.subplots()
ax3.bar(avg_scores.index, avg_scores.values)
ax3.set_xlabel("Score Type")
ax3.set_ylabel("Average Score")
ax3.set_title("Average Candidate Scores")

st.pyplot(fig3)

st.subheader("Skill Distribution")

skill_count = {
    "Python": df["Python"].sum(),
    "SQL": df["SQL"].sum(),
    "Machine Learning": df["MachineLearning"].sum()
}

fig4, ax4 = plt.subplots()
ax4.bar(skill_count.keys(), skill_count.values())
ax4.set_xlabel("Skills")
ax4.set_ylabel("Number of Candidates")
ax4.set_title("Skill Distribution")

st.pyplot(fig4)
ax4.bar(skill_count.keys(), skill_count.values())
ax4.set_xlabel("Skills")
ax4.set_ylabel("Number of Candidates")
ax4.set_title("Detected Skill Count")

st.pyplot(fig4)

st.subheader("Top Candidates")

top_candidates = df.sort_values(
    by="PerformanceScore",
    ascending=False
).head(5)

st.dataframe(top_candidates)