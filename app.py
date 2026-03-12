import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_data.csv")

# Features and target
X = df[["study_hours", "attendance", "assignments"]]
y = df["score"]

# Train model
model = LinearRegression()
model.fit(X, y)

st.title("Student Performance Predictor")

st.write("Predict a student's exam score based on study habits.")

# User inputs
study_hours = st.slider("Study Hours", 0, 10, 5)
attendance = st.slider("Attendance (%)", 0, 100, 70)
assignments = st.slider("Assignments Completed", 0, 10, 5)

if st.button("Predict Score"):

    input_data = [[study_hours, attendance, assignments]]

    prediction = model.predict(input_data)[0]

    # Limit prediction between 0 and 100
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Score: {prediction:.2f}")