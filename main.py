import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from modules.pdf_reader import extract_text_from_pdf
from modules.lab_extractor import extract_lab_values
from modules.health_score import calculate_health_score
from modules.chatbot import chatbot_response
from modules.health_explainer import explain_lab_results
from modules.doctor_questions import doctor_questions

st.set_page_config(page_title="AI Health Educator")

st.title("🩺 AI Personalized Health Educator")

st.write("Understand your health reports and predict disease risks.")

st.warning("This system provides educational insights only and not medical advice.")

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

@st.cache_data
def load_diabetes_model():

    data = pd.read_csv("datasets/diabetes.csv")

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model


@st.cache_data
def load_heart_model():

    data = pd.read_csv("datasets/heart.csv")

    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model


diabetes_model = load_diabetes_model()
heart_model = load_heart_model()

# ------------------------------------------------
# PDF REPORT ANALYSIS
# ------------------------------------------------

st.header("📂 Upload Medical Report")

uploaded_file = st.file_uploader("Upload report", type=["pdf"])

if uploaded_file is not None:

    text = extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted Text")
    st.write(text)

    lab_values = extract_lab_values(text)

    st.subheader("Detected Lab Values")
    st.write(lab_values)

    if lab_values:

        # Plain English Explanation
        explanations = explain_lab_results(lab_values)

        st.subheader("🧠 Plain English Explanation")

        for e in explanations:
            st.write("•", e)

        # Questions for Doctor
        questions = doctor_questions(lab_values)

        st.subheader("❓ Questions You May Ask Your Doctor")

        for q in questions:
            st.write("•", q)

        # Lifestyle Tips
        st.subheader("💡 Lifestyle Tips")

        tips = [
            "Exercise at least 30 minutes daily",
            "Reduce sugar and processed foods",
            "Eat more fruits and vegetables",
            "Maintain healthy body weight",
            "Drink enough water",
            "Sleep at least 7 hours daily"
        ]

        for tip in tips:
            st.write("•", tip)

# ------------------------------------------------
# DIABETES PREDICTION
# ------------------------------------------------

st.header("🔬 Diabetes Risk Prediction")

preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 0, 120)

if st.button("Predict Diabetes Risk"):

    values = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]

    prediction = diabetes_model.predict(values)

    score = calculate_health_score(glucose, bmi, age)

    st.subheader("Health Score")
    st.write(f"{score} / 100")

    if prediction[0] == 1:
        st.error("High risk of diabetes")
    else:
        st.success("Low risk of diabetes")

    # Visualization
    st.subheader("Health Indicators")

    chart_data = {
        "Glucose": glucose,
        "BMI": bmi,
        "Blood Pressure": bp
    }

    fig, ax = plt.subplots()
    ax.bar(chart_data.keys(), chart_data.values())
    st.pyplot(fig)

# ------------------------------------------------
# HEART DISEASE PREDICTION
# ------------------------------------------------

st.header("❤️ Heart Disease Risk Prediction")

age_h = st.number_input("Age", 1, 120, key="heart_age")
sex = st.selectbox("Sex (1=Male, 0=Female)", [1,0])
cp = st.number_input("Chest Pain Type", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.number_input("Rest ECG", 0, 2)
thalach = st.number_input("Max Heart Rate", 60, 220)
exang = st.selectbox("Exercise Angina", [0,1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0)
slope = st.number_input("Slope", 0, 2)
ca = st.number_input("Number of Major Vessels", 0, 3)
thal = st.number_input("Thal", 0, 3)

if st.button("Predict Heart Disease"):

    values = [[age_h,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

    prediction = heart_model.predict(values)

    if prediction[0] == 1:
        st.error("High risk of heart disease")
    else:
        st.success("Low risk of heart disease")

# ------------------------------------------------
# HEALTH CHATBOT
# ------------------------------------------------

st.header("💬 Health Chatbot")

question = st.text_input("Ask a health question")

if st.button("Ask AI"):

    response = chatbot_response(question)

    st.write("AI:", response)