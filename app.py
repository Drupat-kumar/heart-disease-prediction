import streamlit as st
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

@st.cache_resource
def load_artifacts():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    columns = joblib.load("heart_columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_artifacts()

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's medical attributes below to estimate heart disease risk.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                                   format_func=lambda x: "Yes" if x == 1 else "No")
        max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)

    with col2:
        sex = st.selectbox("Sex", options=["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"])
        resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise-Induced Angina", options=["N", "Y"])
        st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict")

if submitted:
    data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope,
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

    if prediction == 1:
        st.error(f"### {result}")
    else:
        st.success(f"### {result}")

    # Build PDF in memory (no disk writes — safer for Streamlit Cloud's read-only-ish filesystem)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Heart Disease Prediction Result", styles["Title"]),
        Spacer(1, 0.5 * inch),
        Paragraph(f"Result: {result}", styles["Normal"]),
    ]
    doc.build(elements)
    buffer.seek(0)

    st.download_button(
        label="Download Result as PDF",
        data=buffer,
        file_name="prediction_result.pdf",
        mime="application/pdf",
    )

st.caption("This tool provides a statistical estimate only and is not a medical diagnosis. Consult a healthcare professional for medical advice.")
