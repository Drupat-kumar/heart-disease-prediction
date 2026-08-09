import streamlit as st
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import math

st.set_page_config(page_title="Cardiac Risk Console", page_icon="🫀", layout="centered")

# ----------------------------------------------------------------------------
# Data / model
# ----------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    columns = joblib.load("heart_columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_artifacts()

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#060A14;
  --panel:rgba(18,24,38,0.55);
  --panel-border:rgba(255,255,255,0.08);
  --text:#EDF1FA;
  --muted:#8993AC;
  --crimson:#FF4568;
  --mint:#21E6A6;
  --amber:#FFB454;
}

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(60% 45% at 15% -5%, rgba(255,69,104,0.16) 0%, rgba(6,10,20,0) 60%),
    radial-gradient(55% 40% at 100% 0%, rgba(33,230,166,0.13) 0%, rgba(6,10,20,0) 60%),
    var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"]{ background: transparent; }
footer{ visibility: hidden; }
.block-container{ padding-top: 2.2rem; max-width: 780px; }

/* ---------- Hero ---------- */
.hero-wrap{ text-align:center; margin-bottom: 0.4rem; }
.eyebrow{
  font-family:'JetBrains Mono', monospace;
  font-size: 0.72rem; letter-spacing: 0.22em; color: var(--muted);
  text-transform: uppercase;
}
.hero-title{
  font-family:'Space Grotesk', sans-serif; font-weight:700;
  font-size: 2.3rem; line-height:1.15; margin: 0.35rem 0 0.1rem 0;
  background: linear-gradient(90deg, #EDF1FA 30%, #9AB2FF 100%);
  -webkit-background-clip: text; background-clip:text; color:transparent;
  text-shadow: 0 0 34px rgba(120,150,255,0.25);
}
.hero-sub{ color: var(--muted); font-size:0.92rem; margin-bottom: 1.1rem; }

.ecg-wrap{
  width:100%; overflow:hidden; height:56px; margin: 0.4rem 0 1.4rem 0;
  -webkit-mask-image: linear-gradient(90deg, transparent, black 12%, black 88%, transparent);
  mask-image: linear-gradient(90deg, transparent, black 12%, black 88%, transparent);
}
.ecg-track{ display:flex; width:2000px; animation: ecgScroll linear infinite; }
.ecg-track svg{ display:block; }
@keyframes ecgScroll{ from{ transform: translateX(0); } to{ transform: translateX(-1000px); } }

/* ---------- Section labels ---------- */
.section-label{
  font-family:'JetBrains Mono', monospace; font-size:0.72rem; letter-spacing:0.18em;
  color: var(--amber); text-transform:uppercase; margin: 0.2rem 0 0.6rem 2px;
  display:flex; align-items:center; gap:0.5rem;
}
.section-label::before{
  content:''; width:7px; height:7px; border-radius:50%;
  background: var(--amber); box-shadow: 0 0 10px var(--amber);
}

/* ---------- Glass form card ---------- */
[data-testid="stForm"]{
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 20px;
  padding: 1.8rem 1.8rem 1.4rem 1.8rem;
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow:
    0 1px 0 rgba(255,255,255,0.06) inset,
    0 30px 60px rgba(0,0,0,0.45),
    0 0 0 1px rgba(255,255,255,0.02);
  transition: transform 0.35s ease, box-shadow 0.35s ease;
  transform: perspective(1200px) rotateX(0deg);
}
[data-testid="stForm"]:hover{
  transform: perspective(1200px) rotateX(0.6deg) translateY(-2px);
  box-shadow:
    0 1px 0 rgba(255,255,255,0.08) inset,
    0 40px 80px rgba(0,0,0,0.55);
}

/* Labels & inputs */
[data-testid="stWidgetLabel"] p{
  font-family:'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important; letter-spacing:0.08em;
  color: var(--muted) !important; text-transform: uppercase;
}
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family:'JetBrains Mono', monospace !important;
}
[data-testid="stNumberInput"] input:focus{
  border-color: var(--mint) !important;
  box-shadow: 0 0 0 1px var(--mint) !important;
}

/* Predict button — heartbeat pulse */
[data-testid="stFormSubmitButton"] button{
  width:100%; margin-top: 0.6rem;
  background: linear-gradient(120deg, var(--crimson), #FF7A8A);
  border: none; border-radius: 999px;
  color: #060A14; font-weight: 700; letter-spacing:0.04em;
  padding: 0.7rem 0; font-family:'Space Grotesk', sans-serif;
  box-shadow: 0 0 0 0 rgba(255,69,104,0.55);
  animation: pulseGlow 2.2s ease-in-out infinite;
  transition: transform 0.15s ease;
}
[data-testid="stFormSubmitButton"] button:hover{ transform: translateY(-1px) scale(1.01); }
@keyframes pulseGlow{
  0%   { box-shadow: 0 0 0 0 rgba(255,69,104,0.45); }
  60%  { box-shadow: 0 0 0 14px rgba(255,69,104,0); }
  100% { box-shadow: 0 0 0 0 rgba(255,69,104,0); }
}

/* Result monitor card */
.monitor-marker + div[data-testid="stVerticalBlockBorderWrapper"]{
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 20px;
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  box-shadow: 0 30px 60px rgba(0,0,0,0.45);
  animation: cardIn 0.55s cubic-bezier(.2,.8,.2,1);
}
@keyframes cardIn{ from{ opacity:0; transform: translateY(10px) scale(0.985);} to{opacity:1; transform:none;} }

.result-flex{ display:flex; align-items:center; gap: 1.6rem; padding: 0.4rem 0.4rem; }
.gauge-num{
  font-family:'JetBrains Mono', monospace; font-weight:600;
  font-size: 1.5rem; fill: var(--text);
}
.result-text-title{
  font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1.3rem; margin:0 0 0.15rem 0;
}
.result-text-sub{ color: var(--muted); font-size:0.85rem; font-family:'JetBrains Mono', monospace; }

[data-testid="stDownloadButton"] button{
  background: transparent; border: 1px solid var(--mint); color: var(--mint);
  border-radius: 999px; font-family:'Space Grotesk', sans-serif; font-weight:600;
  width:100%; margin-top: 1rem;
}
[data-testid="stDownloadButton"] button:hover{ background: rgba(33,230,166,0.1); }

.disclaimer{
  text-align:center; color: var(--muted); font-size:0.78rem;
  font-family:'JetBrains Mono', monospace; margin-top:2rem; opacity:0.75;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def ecg_svg_html(color: str, duration: str, height: int = 56):
    """One PQRST-like tile repeated, scrolled seamlessly via CSS translateX."""
    tile = ("M0,28 L34,28 L42,10 L50,46 L58,28 L94,28 "
            "L102,20 L110,28 L200,28")
    tiles = "".join(
        f'<path d="{tile}" transform="translate({i*200},0)" '
        f'stroke="{color}" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        for i in range(10)
    )
    return f"""
    <div class="ecg-wrap" style="height:{height}px;">
      <div class="ecg-track" style="animation-duration:{duration};">
        <svg width="2000" height="{height}" viewBox="0 0 2000 56" xmlns="http://www.w3.org/2000/svg">
          {tiles}
        </svg>
      </div>
    </div>
    """


def gauge_svg_html(prob: float, color: str):
    """Radial risk gauge driven by the model's actual predict_proba output."""
    r, cx, cy, sw = 60, 70, 70, 12
    circumference = 2 * math.pi * r
    offset = circumference * (1 - prob)
    pct = round(prob * 100)
    return f"""
    <svg width="140" height="140" viewBox="0 0 140 140">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="{sw}"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}"
        stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
        transform="rotate(-90 {cx} {cy})" style="filter:drop-shadow(0 0 8px {color});"/>
      <text x="{cx}" y="{cy+7}" text-anchor="middle" class="gauge-num">{pct}%</text>
    </svg>
    """


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-wrap">
      <div class="eyebrow">Vitals · Diagnostics · Risk Model</div>
      <div class="hero-title">Cardiac Risk Console</div>
      <div class="hero-sub">Enter a patient profile to estimate heart disease risk from clinical attributes.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(ecg_svg_html("#5C7CFA", "5.5s"), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Form
# ----------------------------------------------------------------------------

with st.form("prediction_form"):
    st.markdown('<div class="section-label">Vitals</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        fasting_bs = st.selectbox("Fasting Sugar > 120", options=[0, 1],
                                   format_func=lambda x: "Yes" if x == 1 else "No")
    with c2:
        resting_bp = st.number_input("Resting BP", min_value=0, max_value=250, value=120)
        max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    with c3:
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
        oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)

    st.markdown('<div class="section-label" style="margin-top:0.9rem;">Clinical Profile</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        sex = st.selectbox("Sex", options=["M", "F"])
        st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])
    with c5:
        chest_pain = st.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"])
    with c6:
        resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise Angina", options=["N", "Y"])

    submitted = st.form_submit_button("⚡  RUN PREDICTION")

# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------

if submitted:
    data = {
        "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "MaxHR": max_hr, "Oldpeak": oldpeak,
        "Sex": sex, "ChestPainType": chest_pain, "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina, "ST_Slope": st_slope,
    }
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0]
    risk_prob = float(proba[list(model.classes_).index(1)])

    high_risk = prediction == 1
    color = "#FF4568" if high_risk else "#21E6A6"
    result_title = "High Risk of Heart Disease" if high_risk else "Low Risk of Heart Disease"
    ecg_speed = "1.4s" if high_risk else "4.5s"

    st.markdown('<div class="monitor-marker"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-flex">
              {gauge_svg_html(risk_prob, color)}
              <div>
                <div class="result-text-title" style="color:{color};">{result_title}</div>
                <div class="result-text-sub">MODEL CONFIDENCE · {risk_prob*100:.1f}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(ecg_svg_html(color, ecg_speed, height=46), unsafe_allow_html=True)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("Heart Disease Prediction Result", styles["Title"]),
            Spacer(1, 0.5 * inch),
            Paragraph(f"Result: {result_title}", styles["Normal"]),
            Paragraph(f"Model confidence: {risk_prob*100:.1f}%", styles["Normal"]),
        ]
        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            label="Download Result as PDF",
            data=buffer,
            file_name="prediction_result.pdf",
            mime="application/pdf",
        )

st.markdown(
    '<div class="disclaimer">Statistical estimate only — not a medical diagnosis. Consult a clinician.</div>',
    unsafe_allow_html=True,
)
