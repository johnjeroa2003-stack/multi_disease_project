import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")
st.title("🧑‍⚕️ Multiple Disease Prediction System")
st.write("Predict **Kidney Disease**, **Heart Disease**, and **Diabetes**")

# ✅ function to load model
def load_model(model_name):
    data = pickle.load(open(f"models/{model_name}.pkl", "rb"))
    return data["model"], data["scaler"], data["feature_names"]


# Page selection
choice = st.selectbox(
    "Select Disease to Predict",
    ["Kidney Disease", "Diabetes", "Heart Disease"]
)


# -------------------- KIDNEY UI --------------------
if choice == "Kidney Disease":
    st.header("🟡 Kidney Disease Prediction")

    age = st.number_input("Age", 1, 120)
    bp = st.number_input("Blood Pressure")
    sg = st.number_input("Specific Gravity")

    if st.button("Predict Kidney Disease"):

        model, scaler, feature_names = load_model("kidney_model")

        df = pd.DataFrame([[age, bp, sg]], columns=["age", "bp", "sg"])
        df = df.reindex(columns=feature_names, fill_value=0)

        X = scaler.transform(df)
        result = model.predict(X)[0]

        st.success("✅ NO Chronic Kidney Disease") if result == "notckd" else st.error("⚠ CKD DETECTED")


# -------------------- DIABETES UI --------------------
elif choice == "Diabetes":
    st.header("🟢 Diabetes Prediction")

    preg = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level")
    bmi = st.number_input("BMI")

    if st.button("Predict Diabetes"):

        model, scaler, feature_names = load_model("diabetes_model")

        df = pd.DataFrame([[preg, glucose, bmi]], columns=["Pregnancies", "Glucose", "BMI"])
        df = df.reindex(columns=feature_names, fill_value=0)

        X = scaler.transform(df)
        result = model.predict(X)[0]

        st.success("✅ No Diabetes") if result == 0 else st.error("⚠ Diabetes Detected")


# -------------------- HEART UI --------------------
elif choice == "Heart Disease":
    st.header("🔴 Heart Disease Prediction")

    age = st.number_input("Age", 1, 120)
    chol = st.number_input("Cholesterol")
    trestbps = st.number_input("Resting Blood Pressure")

    if st.button("Predict Heart Disease"):

        model, scaler, feature_names = load_model("heart_model")

        df = pd.DataFrame([[age, chol, trestbps]], columns=["age", "chol", "trestbps"])
        df = df.reindex(columns=feature_names, fill_value=0)

        X = scaler.transform(df)
        result = model.predict(X)[0]

        st.success("✅ Heart is Healthy") if result == 0 else st.error("⚠ Heart Disease Risk")
