from flask import Flask, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# ----------------------------
# Load trained model files
# ----------------------------
model = joblib.load("disease_model.pkl")
mlb = joblib.load("symptom_encoder.pkl")

# Load dataset for medicine lookup
df = pd.read_csv("disease_dataset_3000_rows.csv")


@app.route("/")
def home():
    return "AI Healthcare Backend is Running Successfully 🚀"

# 🔴 Define serious diseases
SERIOUS_DISEASES = [
    "Diabetes",
    "Hypertension",
    "Asthma",
    "Tuberculosis",
    "Hepatitis A",
    "Hepatitis B",
    "Dengue",
    "Malaria",
    "Heart Disease",
    "COPD",
    "Liver Cirrhosis",
    "Pneumonia",
    "Influenza (Severe)",
    "COVID-19"
]

# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        symptoms = data.get("symptoms", [])
        age = data.get("age", 0)
        allergies = data.get("allergies", [])

        if not isinstance(symptoms, list):
            return "Error: Symptoms must be a list", 400

        # Encode symptoms
        input_encoded = mlb.transform([symptoms])

        # Convert to DataFrame
        input_df = pd.DataFrame(input_encoded)
        input_df["Age"] = age
        input_df.columns = input_df.columns.astype(str)

        # Get Top 3 Predictions
        probabilities = model.predict_proba(input_df)[0]
        top3_idx = np.argsort(probabilities)[-3:][::-1]

        results = []
        emergency_flag = False

        for idx in top3_idx:
            disease_name = model.classes_[idx]
            confidence_value = round(probabilities[idx] * 100, 2)

            # Get medicines
            disease_row = df[df["Disease"] == disease_name]

            if disease_row.empty:
                medicines_list = []
            else:
                medicines = disease_row["Medicines"].values[0]
                medicines_list = [m.strip() for m in medicines.split(",")]

            # Remove allergic medicines
            allergies_lower = [a.lower() for a in allergies]

            final_medicines = [
                med for med in medicines_list
                if med.lower() not in allergies_lower
            ]

            # Emergency condition
            if confidence_value > 75 and disease_name in SERIOUS_DISEASES:
                emergency_flag = True

            results.append({
                "disease": disease_name,
                "confidence": f"{confidence_value}%",
                "recommended_medicines": final_medicines
            })

        # ----------------------------
        # Generate Professional Report
        # ----------------------------

        report = "\n🩺 AI HEALTHCARE DIAGNOSTIC REPORT\n"
        report += "=============================\n\n"

        report += f"Patient Age: {age}\n"
        report += f"Reported Symptoms: {', '.join(symptoms)}\n\n"

        report += "Top 3 Possible Conditions:\n"
        report += "---------------------------------\n\n"

        for i, item in enumerate(results, start=1):
            report += f"{i}. Disease: {item['disease']}\n"
            report += f"   Confidence Level: {item['confidence']}\n"

            if item["recommended_medicines"]:
                report += f"   Recommended Medicines: {', '.join(item['recommended_medicines'])}\n\n"
            else:
                report += "   Recommended Medicines: No safe medicines available (check allergies)\n\n"

        # Emergency Alert
        if emergency_flag:
            report += "⚠ EMERGENCY ALERT:\n"
            report += "Immediate medical consultation is strongly advised.\n\n"

        report += "---------------------------------------\n"
        report += "Disclaimer: This AI system provides preliminary guidance only.\n"
        report += "Please consult a licensed medical professional for diagnosis.\n"

        return report, 200, {'Content-Type': 'text/plain'}

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

    