
from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import os

app = Flask(__name__)

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
columns = joblib.load("heart_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = {
            "Age": int(request.form["Age"]),
            "RestingBP": int(request.form["RestingBP"]),
            "Cholesterol": int(request.form["Cholesterol"]),
            "FastingBS": int(request.form["FastingBS"]),
            "MaxHR": int(request.form["MaxHR"]),
            "Oldpeak": float(request.form["Oldpeak"]),
            "Sex": request.form["Sex"],
            "ChestPainType": request.form["ChestPainType"],
            "RestingECG": request.form["RestingECG"],
            "ExerciseAngina": request.form["ExerciseAngina"],
            "ST_Slope": request.form["ST_Slope"]
        }

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
        df_scaled = scaler.transform(df)

        prediction = model.predict(df_scaled)[0]
        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

        # Create PDF
        pdf_path = "prediction_result.pdf"
        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        elements = []
        elements.append(Paragraph("Heart Disease Prediction Result", styles["Title"]))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"Result: {result}", styles["Normal"]))
        doc.build(elements)

        return render_template("result.html", result=result)

    return render_template("index.html")

@app.route("/download")
def download():
    return send_file("prediction_result.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
