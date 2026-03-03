import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("../data/disease_dataset_3000_rows.csv")

# ----------------------------
# Preprocess Symptoms
# ----------------------------
df['Symptoms'] = df['Symptoms'].apply(lambda x: x.split(","))

mlb = MultiLabelBinarizer()
symptoms_encoded = mlb.fit_transform(df['Symptoms'])

# ----------------------------
# Add Age as Feature
# ----------------------------
X = pd.DataFrame(symptoms_encoded)
X['Age'] = df['Age']

# 🔥 FIX: Convert all column names to string
X.columns = X.columns.astype(str)

y = df['Disease']

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Model
# ----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# ----------------------------
# Save Model
# ----------------------------
pickle.dump(model, open("disease_model.pkl", "wb"))
pickle.dump(mlb, open("symptom_encoder.pkl", "wb"))

print("Model saved successfully!")