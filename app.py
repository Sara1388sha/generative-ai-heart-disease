# ==============================
# Generative AI for Heart Disease Education
# Educational & Research Purposes Only
# ==============================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ollama
import gradio as gr

# ------------------------------
# Load Data
# ------------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

df = pd.read_csv(url, header=None, names=columns, na_values="?")
df = df.dropna()
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

X = df.drop(columns=["num", "target"])
y = df["target"]

# ------------------------------
# Train ML Model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Helper Functions
# ------------------------------
def create_profile(row):
    return f"""
Age: {int(row['age'])}
Cholesterol: {row['chol']}
Blood Pressure: {row['trestbps']}
Max Heart Rate: {row['thalach']}
"""

def call_llm(prompt, system_prompt):
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]

SYSTEM_PROMPT = """
You are a medical assistant.
Give safe, general, evidence-based advice.
Always recommend consulting a healthcare professional.
"""

# ------------------------------
# ML + LLM Pipeline
# ------------------------------
def heart_risk_pipeline():
    sample = df.iloc[0]

    profile = create_profile(sample)

    X_sample = pd.DataFrame([sample[X.columns]])
    risk_prob = model.predict_proba(X_sample)[0][1]

    risk_level = "HIGH risk" if risk_prob >= 0.3 else "LOW risk"

    prompt = f"""
Patient Profile:
{profile}

Predicted Risk Probability: {risk_prob:.2f}
Risk Level: {risk_level}

Provide general lifestyle advice for heart health.
"""

    response = call_llm(prompt, SYSTEM_PROMPT)

    return profile, risk_prob, risk_level, response

# ------------------------------
# Run Example
# ------------------------------
if __name__ == "__main__":
    profile, prob, risk, advice = heart_risk_pipeline()

    print("=== PATIENT PROFILE ===")
    print(profile)

    print(f"Predicted Risk Probability: {prob:.2f}")
    print("Risk Level:", risk)

    print("\n=== AI ADVICE ===")
    print(advice)


def medical_chat(user_input):
    return call_llm(user_input, SYSTEM_PROMPT)

interface = gr.Interface(
    fn=medical_chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask a medical question..."),
    outputs="text",
    title="Medical AI Assistant 🩺",
    description="Educational use only. Not medical advice."
)

# Uncomment for demo
# interface.launch()
