💼 Salary Prediction System

A machine learning–based system to predict expected salaries for NYC government job postings using structured job attributes and unstructured text data. The project covers the complete data science lifecycle, from data preprocessing and model training to deployment using Streamlit.

📌 Project Overview

Salary estimation is an important aspect of workforce planning and transparency. This project aims to automate salary prediction by analyzing historical job posting data and learning patterns based on job category, career level, agency, employment type, and job descriptions.

The final solution includes:

A trained and tuned Random Forest regression model

A Streamlit web application for real-time salary prediction

A Power BI dashboard for salary trend analysis

**🎯 Problem Statement**

To build a machine learning model that predicts the expected salary for NYC government job postings based on job-related attributes and descriptions.

**🧾 Dataset**

Source: NYC Government Job Postings

Type: Real-world dataset with structured and unstructured features

Target Variable:

salary_mid = (Salary Range From + Salary Range To) / 2

⚠️ The dataset is not included in this repository due to GitHub file size limitations.

🛠️ Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, SciPy

Model: Random Forest Regressor

Deployment: Streamlit

Visualization: Power BI

**🔄 Project Workflow**

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Model Training (Linear Regression, Random Forest, Gradient Boosting)

Model Evaluation (RMSE, R², Accuracy ±10%)

Model Selection & Saving

Deployment using Streamlit

**🤖 Machine Learning Models**

The following models were trained and evaluated:

Model	Description
Linear Regression	Baseline regression model
Random Forest Regressor	Final selected model
Gradient Boosting Regressor	Ensemble-based model
✅ Final Model Selection

Random Forest Regressor was selected due to:

Lowest RMSE

Highest R² score

Best tolerance-based accuracy

**💼 Salary Prediction System**


Since this is a regression problem, the following metrics were used:

RMSE (Root Mean Squared Error)

R² Score

Accuracy (±10%) – percentage of predictions within ±10% of actual salary

**🚀 Deployment**

The final model is deployed using Streamlit, allowing users to:

Select job attributes using dropdowns

Enter job descriptions

Receive predicted salary along with an expected range

The Streamlit app uses a pre-trained model and does not require the original dataset.

  📂 Project Structure
salary-prediction-system/
│
├── app.py                       # Streamlit application
├── main.py                      # Training, evaluation & model saving
├── salary_prediction_rf_model.pkl
├── requirements.txt
├── README.md

               ▶️ How to Run the Application
               1️⃣ Install dependencies
                pip install -r requirements.txt

               2️⃣ Run Streamlit app
                streamlit run app.py

📈 Power BI Dashboard

A Power BI dashboard was created to analyze:

Salary trends by job category and career level

Full-time vs part-time salary comparison

Top agencies by salary

Salary distribution and variability

This complements the machine learning model with business-focused insights.

**🔮 Future Enhancements**
Integrate advanced NLP models

Automate hyperparameter tuning

Add explainability using SHAP values

Expand dataset to other regions

Improve UI and add confidence intervals

👨‍💻 Developed By

Farhan
Machine Learning / Data Science 
