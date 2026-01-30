import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💼",
    layout="centered"
)

# --------------------------------------------------
# Load Dataset (for dropdown values)
# --------------------------------------------------
df = pd.read_csv("Jobs_NYC_Postings.csv")

job_categories = sorted(df["Job Category"].dropna().unique())
career_levels = sorted(df["Career Level"].dropna().unique())
agencies = sorted(df["Agency"].dropna().unique())

# --------------------------------------------------
# Load Model Artifacts
# --------------------------------------------------
with open("salary_prediction_rf_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
ohe = artifacts["ohe"]
tfidf = artifacts["tfidf"]

num_features = artifacts["num_features"]
cat_features = artifacts["cat_features"]
text_feature = artifacts["text_feature"]

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>💼 Salary Prediction System</h1>
    <p style='text-align: center; color: gray;'>
    Predict expected salary for NYC government job postings
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("🔍 Enter Job Details")

col1, col2 = st.columns(2)

with col1:
    positions = st.number_input(
        "Number of Positions",
        min_value=1,
        max_value=100,
        value=1
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Full-Time", "Part-Time"]
    )

    job_category = st.selectbox(
        "Job Category",
        job_categories
    )

with col2:
    career_level = st.selectbox(
        "Career Level",
        career_levels
    )

    agency = st.selectbox(
        "Agency",
        agencies
    )

job_text = st.text_area(
    "Job Description / Skills / Qualifications",
    height=180,
    placeholder="Enter job responsibilities, required skills, and qualifications..."
)

is_full_time = 1 if employment_type == "Full-Time" else 0

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown("")

if st.button("💰 Predict Salary", use_container_width=True):

    input_df = pd.DataFrame({
        "# Of Positions": [positions],
        "is_full_time": [is_full_time],
        "Job Category": [job_category],
        "Career Level": [career_level],
        "Agency_clean": [agency],
        "text_combined": [job_text]
    })

    X_num = input_df[num_features].values
    X_cat = ohe.transform(input_df[cat_features])
    X_txt = tfidf.transform(input_df[text_feature])

    X_final = hstack([X_num, X_cat, X_txt])

    prediction = model.predict(X_final.toarray())[0]

    lower = prediction * 0.9
    upper = prediction * 1.1

    st.success("### 🎯 Salary Prediction Result")
    st.metric(
        label="Estimated Salary",
        value=f"₹ {prediction:,.2f}"
    )

    st.write(
        f"**Expected Range:** ₹ {lower:,.2f}  —  ₹ {upper:,.2f}"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Internship Project — Salary Prediction System</p>
        <p><strong>Developed by Farhan</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)
