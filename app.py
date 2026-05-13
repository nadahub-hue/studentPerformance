pip install streamlit pandas sckit-learn
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="UTAS Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 UTAS Student Performance Prediction")
st.write("This UI predicts a student's exam score based on lifestyle and study habits.")

DATA_FILES = [
    "UTAS_students_performance.csv",
    "student_habits_performance(1).csv"
]

@st.cache_data
def load_data():
    for file in DATA_FILES:
        path = Path(file)
        if path.exists():
            return pd.read_csv(path), file
    st.error(
        "No CSV file found. Please put this app.py in the same folder as the dataset CSV file."
    )
    st.stop()

df, used_file = load_data()
st.success(f"Dataset loaded: {used_file}")

target_col = "exam_score"

if target_col not in df.columns:
    st.error("The dataset must contain an 'exam_score' column.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", SVR(kernel="rbf"))
])

test_size = 0.2 if len(df) >= 30 else 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

model.fit(X_train, y_train)

with st.expander("📊 Model Evaluation"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): **{mse:.2f}**")
    st.write(f"R² Score: **{r2:.2f}**")

st.subheader("Enter Student Information")

user_input = {}

for col in X.columns:
    if col in numeric_features:
        min_value = float(df[col].min())
        max_value = float(df[col].max())
        mean_value = float(df[col].mean())

        if min_value == max_value:
            max_value = min_value + 1

        user_input[col] = st.slider(
            label=col.replace("_", " ").title(),
            min_value=min_value,
            max_value=max_value,
            value=mean_value
        )
    else:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        if not options:
            options = ["Unknown"]
        user_input[col] = st.selectbox(
            label=col.replace("_", " ").title(),
            options=options
        )

input_df = pd.DataFrame([user_input])

if st.button("Predict Exam Score"):
    prediction = model.predict(input_df)[0]
    prediction = max(0, min(100, prediction))

    st.subheader("Prediction Result")
    st.metric("Predicted Exam Score", f"{prediction:.2f} / 100")

    if prediction >= 80:
        st.success("High performance prediction.")
    elif prediction >= 60:
        st.warning("Medium performance prediction.")
    else:
        st.error("Low performance prediction.")

st.caption("Deployment UI created using Streamlit.")
