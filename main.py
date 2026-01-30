import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from scipy.sparse import hstack

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("Jobs_NYC_Postings.csv")

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------
df = df.dropna(subset=["Salary Range From", "Salary Range To"])

df["# Of Positions"] = df["# Of Positions"].fillna(1)

cat_cols = ["Job Category", "Career Level",
            "Full-Time/Part-Time indicator", "Agency"]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

text_cols = ["Job Description",
             "Minimum Qual Requirements",
             "Preferred Skills"]
for col in text_cols:
    df[col] = df[col].fillna("")

df["salary_mid"] = (df["Salary Range From"] + df["Salary Range To"]) / 2

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
df["text_combined"] = (
    df["Job Description"] + " " +
    df["Minimum Qual Requirements"] + " " +
    df["Preferred Skills"]
)

df["is_full_time"] = df["Full-Time/Part-Time indicator"].apply(
    lambda x: 1 if str(x).lower().startswith("f") else 0
)

top_agencies = df["Agency"].value_counts()
top_agencies = top_agencies[top_agencies >= 50].index

df["Agency_clean"] = df["Agency"].apply(
    lambda x: x if x in top_agencies else "Other"
)

num_features = ["# Of Positions", "is_full_time"]
cat_features = ["Job Category", "Career Level", "Agency_clean"]
text_feature = "text_combined"

X = df[num_features + cat_features + [text_feature]]
y = df["salary_mid"]

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Encoding
# --------------------------------------------------
X_train_num = X_train[num_features]
X_test_num = X_test[num_features]

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[cat_features])
X_test_cat = ohe.transform(X_test[cat_features])

tfidf = TfidfVectorizer(
    max_features=6000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_txt = tfidf.fit_transform(X_train[text_feature])
X_test_txt = tfidf.transform(X_test[text_feature])

X_train_final = hstack([X_train_num.values,
                         X_train_cat,
                         X_train_txt])

X_test_final = hstack([X_test_num.values,
                        X_test_cat,
                        X_test_txt])

# --------------------------------------------------
# Models
# --------------------------------------------------
lr = LinearRegression()

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingRegressor(random_state=42)

# --------------------------------------------------
# Training
# --------------------------------------------------
lr.fit(X_train_final, y_train)
rf.fit(X_train_final.toarray(), y_train)
gb.fit(X_train_final.toarray(), y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred_lr = lr.predict(X_test_final)
y_pred_rf = rf.predict(X_test_final.toarray())
y_pred_gb = gb.predict(X_test_final.toarray())

# --------------------------------------------------
# Accuracy Function (±10%)
# --------------------------------------------------
def regression_accuracy(y_true, y_pred, tolerance=0.10):
    lower = y_true * (1 - tolerance)
    upper = y_true * (1 + tolerance)
    return ((y_pred >= lower) & (y_pred <= upper)).mean() * 100

# --------------------------------------------------
# Results
# --------------------------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        np.sqrt(mean_squared_error(y_test, y_pred_gb))
    ],
    "R2": [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_gb)
    ],
    "Accuracy (±10%)": [
        regression_accuracy(y_test, y_pred_lr),
        regression_accuracy(y_test, y_pred_rf),
        regression_accuracy(y_test, y_pred_gb)
    ]
})

print("\nFINAL MODEL RESULTS\n")
print(results.round(3))

# --------------------------------------------------
# Save BEST MODEL (Random Forest)
# --------------------------------------------------
model_artifacts = {
    "model": rf,
    "ohe": ohe,
    "tfidf": tfidf,
    "num_features": num_features,
    "cat_features": cat_features,
    "text_feature": text_feature
}

with open("salary_prediction_rf_model.pkl", "wb") as f:
    pickle.dump(model_artifacts, f)

print("\nModel saved as salary_prediction_rf_model.pkl")
