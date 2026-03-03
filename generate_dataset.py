import random
import pandas as pd

# 40 diseases with base symptoms & medicines
disease_data = {
    "Flu": {
        "symptoms": ["fever", "cough", "body_aches", "chills", "fatigue"],
        "medicines": ["Paracetamol", "Oseltamivir", "Ibuprofen"]
    },
    "Common Cold": {
        "symptoms": ["sneezing", "runny_nose", "sore_throat", "nasal_congestion"],
        "medicines": ["Cetirizine", "Paracetamol", "Phenylephrine"]
    },
    "Diabetes": {
        "symptoms": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision"],
        "medicines": ["Metformin", "Insulin", "Glibenclamide"]
    },
    "Hypertension": {
        "symptoms": ["headache", "chest_pain", "shortness_of_breath", "fatigue"],
        "medicines": ["Amlodipine", "Losartan", "Atenolol"]
    },
    "Asthma": {
        "symptoms": ["wheezing", "shortness_of_breath", "chest_tightness", "cough"],
        "medicines": ["Salbutamol", "Budesonide", "Montelukast"]
    },
    "Hepatitis A": {
        "symptoms": ["yellowish_skin", "fatigue", "abdominal_pain", "dark_urine"],
        "medicines": ["Rest", "Antiemetics", "IV Fluids"]
    },
    "Hepatitis B": {
        "symptoms": ["yellowish_skin", "fatigue", "joint_pain", "loss_of_appetite"],
        "medicines": ["Tenofovir", "Entecavir", "Interferon"]
    },
    "Dengue": {
        "symptoms": ["high_fever", "joint_pain", "rash", "headache"],
        "medicines": ["Paracetamol", "IV Fluids", "Electrolytes"]
    },
    "Malaria": {
        "symptoms": ["fever", "chills", "sweating", "headache"],
        "medicines": ["Chloroquine", "Artemether", "Lumefantrine"]
    },
    # Add more diseases similarly...
}

# Add dummy diseases until 40
while len(disease_data) < 40:
    disease_name = f"Disease_{len(disease_data)+1}"
    disease_data[disease_name] = {
        "symptoms": ["fatigue", "pain", "fever", "nausea"],
        "medicines": ["MedicineA", "MedicineB", "MedicineC"]
    }

rows = []

for disease, info in disease_data.items():
    for _ in range(70):  # 70 variations per disease → ~2800 rows
        age = random.randint(5, 80)

        base_symptoms = info["symptoms"]
        selected_symptoms = random.sample(
            base_symptoms,
            random.randint(2, len(base_symptoms))
        )

        symptom_string = ",".join(selected_symptoms)

        rows.append({
            "Age": age,
            "Symptoms": symptom_string,
            "Disease": disease,
            "Medicines": ",".join(info["medicines"])
        })

df = pd.DataFrame(rows)
df.to_csv("../data/disease_dataset.csv", index=False)

print("Large dataset generated successfully!")