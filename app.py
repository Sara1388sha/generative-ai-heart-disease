# =========================================
# Generative AI for Heart Disease Education
# Educational & Research Use Only
# =========================================

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================================
# SAFE OLLAMA IMPORT
# =========================================

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# =========================================
# SYSTEM PROMPT
# =========================================

SYSTEM_PROMPT = """
You are a medical assistant.
Give safe, general, evidence-based advice.
Always recommend consulting a healthcare professional.
"""

# =========================================
# LLM FUNCTION
# =========================================

def call_llm(prompt, system_prompt):

    response = f"""
Based on the patient profile, the predicted heart disease risk is moderate.

Recommendations:
- Maintain a healthy diet
- Exercise regularly
- Reduce cholesterol intake
- Monitor blood pressure
- Avoid smoking
- Consult a healthcare professional
"""

    return response

# =========================================
# MULTILINGUAL PROMPT
# =========================================

def build_multilingual_prompt(profile, risk_prob, lang="en"):

    if lang == "fa":

        return f"""
شما یک دستیار پزشکی هستید.

اطلاعات بیمار:
{profile}

احتمال ریسک:
{risk_prob:.2f}

بر اساس اطلاعات بالا توصیه پزشکی ارائه بده.
حتماً توصیه کن بیمار با پزشک مشورت کند.
"""

    elif lang == "fr":

        return f"""
Vous êtes un assistant médical.

Profil du patient:
{profile}

Probabilité de risque:
{risk_prob:.2f}

Fournissez des conseils médicaux personnalisés.
Recommandez de consulter un médecin.
"""

    else:

        return f"""
You are a medical assistant.

Patient Profile:
{profile}

Predicted Risk Probability:
{risk_prob:.2f}

Provide personalized medical advice.
Always recommend consulting a healthcare professional.
"""

# =========================================
# SAMPLE DATA
# =========================================

data = {
    "age": [45, 54, 37, 62, 50],
    "cholesterol": [220, 180, 190, 260, 210],
    "blood_pressure": [140, 130, 120, 170, 150],
    "target": [1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "cholesterol", "blood_pressure"]]
y = df["target"]

# =========================================
# MODEL TRAINING
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# =========================================
# STREAMLIT UI
# =========================================

st.title("❤️ Generative AI Heart Disease Assistant")

age = st.slider("Age", 20, 90, 50)
cholesterol = st.slider("Cholesterol", 100, 300, 200)
blood_pressure = st.slider("Blood Pressure", 80, 200, 130)

language = st.selectbox(
    "Language",
    ["en", "fa", "fr"]
)

# =========================================
# PREDICTION
# =========================================

if st.button("Predict Risk"):

    sample = pd.DataFrame([{
        "age": age,
        "cholesterol": cholesterol,
        "blood_pressure": blood_pressure
    }])

    risk_prob = model.predict_proba(sample)[0][1]

    prediction = int(risk_prob > 0.5)

    if prediction == 1:
        risk = "HIGH RISK"
    else:
        risk = "LOW RISK"

    st.subheader(f"Prediction: {risk}")
    st.write(f"Risk Probability: {risk_prob:.2f}")

    profile = f"""
Age: {age}
Cholesterol: {cholesterol}
Blood Pressure: {blood_pressure}
"""

    prompt = build_multilingual_prompt(
        profile,
        risk_prob,
        language
    )

    response = call_llm(prompt, SYSTEM_PROMPT)

    st.subheader("AI Medical Advice")
    st.write(response)
