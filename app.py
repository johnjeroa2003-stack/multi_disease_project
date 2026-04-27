import streamlit as st
import pickle
import numpy as np

# Load models
parkinsons_model = pickle.load(open("models/parkinsons_model.pkl", "rb"))
kidney_model = pickle.load(open("models/kidney_model.pkl", "rb"))
liver_model = pickle.load(open("models/liver_model.pkl", "rb"))

# Load scalers
parkinsons_scaler = pickle.load(open("models/parkinsons_scaler.pkl", "rb"))
kidney_scaler = pickle.load(open("models/kidney_scaler.pkl", "rb"))
liver_scaler = pickle.load(open("models/liver_scaler.pkl", "rb"))

# UI
st.title("🩺 Multiple Disease Prediction System")
st.write("This system predicts Parkinson’s, Kidney, and Liver diseases using Machine Learning.")
st.info("⚠ This is only a prediction system and not a medical diagnosis.")

option = st.sidebar.selectbox(
    "Select Disease",
    ["Parkinsons", "Kidney", "Liver"]
)

# -----------------------------
# Parkinson
# -----------------------------
if option == "Parkinsons":

    st.header("Parkinson's Prediction")

    col1, col2 = st.columns(2)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", value=120.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", value=150.0)

    with col2:
        flo = st.number_input("MDVP:Flo(Hz)", value=100.0)
        jitter = st.number_input("Jitter", value=0.005)

    if st.button("Predict Parkinson"):

        num_features = parkinsons_model.n_features_in_

        # 🔥 FIX: use ones instead of zeros
        input_data = np.ones(num_features)

        input_data[0] = fo
        input_data[1] = fhi
        input_data[2] = flo
        input_data[3] = jitter

        input_data = np.array([input_data])

        input_data = parkinsons_scaler.transform(input_data)

        prob = parkinsons_model.predict_proba(input_data)
        confidence = prob[0][1]

        st.subheader("Result")

        if confidence > 0.3:
            st.error("⚠ Parkinson Disease Detected")
        else:
            st.success("✅ No Parkinson Disease")

        st.write(f"Confidence: {confidence*100:.2f}%")
        st.progress(int(confidence * 100))


# -----------------------------
# Kidney
# -----------------------------
elif option == "Kidney":

    st.header("Kidney Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", value=40)
        bp = st.number_input("Blood Pressure", value=80)

    with col2:
        sg = st.number_input("Specific Gravity", value=1.02)
        al = st.number_input("Albumin", value=1)

    if st.button("Predict Kidney"):

        num_features = kidney_model.n_features_in_

       
        input_data = np.ones(num_features)

        input_data[0] = age
        input_data[1] = bp
        input_data[2] = sg
        input_data[3] = al

        input_data = np.array([input_data])

        input_data = kidney_scaler.transform(input_data)

        prob = kidney_model.predict_proba(input_data)
        confidence = prob[0][1]

        st.subheader("Result")

        if confidence > 0.3:
            st.error("⚠ Kidney Disease Detected")
        else:
            st.success("✅ No Kidney Disease")

        st.write(f"Confidence: {confidence*100:.2f}%")
        st.progress(int(confidence * 100))


# -----------------------------
# Liver
# -----------------------------
elif option == "Liver":

    st.header("Liver Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", value=45)
        tb = st.number_input("Total Bilirubin", value=1.0)

    with col2:
        db = st.number_input("Direct Bilirubin", value=0.3)
        ap = st.number_input("Alkaline Phosphotase", value=200)

    if st.button("Predict Liver"):

        num_features = liver_model.n_features_in_

        # 🔥 FIX
        input_data = np.ones(num_features)

        input_data[0] = age
        input_data[1] = tb
        input_data[2] = db
        input_data[3] = ap

        input_data = np.array([input_data])

        input_data = liver_scaler.transform(input_data)

        prob = liver_model.predict_proba(input_data)
        confidence = prob[0][1]

        st.subheader("Result")

        if confidence > 0.3:
            st.error("⚠ Liver Disease Detected")
        else:
            st.success("✅ No Liver Disease")

        st.write(f"Confidence: {confidence*100:.2f}%")
        st.progress(int(confidence * 100))