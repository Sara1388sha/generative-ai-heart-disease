# =========================================
# Generative AI for Heart Disease Education
# Educational & Research Use Only
# =========================================

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================================
# OPTIONAL OLLAMA IMPORT
# =========================================

try:
    import ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Heart Disease AI Assistant",
    page_icon="❤️",
    layout="centered"
)

# =========================================
# SYSTEM PROMPT
# =========================================

SYSTEM_PROMPT = """
You are a medical assistant.
Give safe, general, evidence-based advice.
Always recommend consulting a healthcare professional.
"""

# =========================================
# LOAD REAL HEART DATASET
# =========================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

df = pd.read_csv(
    url,
    header=None,
    names=columns,
    na_values="?"
)

df = df.dropna()

# Binary target
df["target"] = df["num"].apply(
    lambda x: 1 if x > 0 else 0
)

# Features
X = df.drop(columns=["num", "target"])
y = df["target"]

# =========================================
# TRAIN MODEL
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================
# MULTILINGUAL LLM FUNCTION
# =========================================

def call_llm(prompt, lang="en"):

    # =========================================
    # OLLAMA RESPONSE
    # =========================================

    if OLLAMA_AVAILABLE:

        try:

            response = ollama.chat(
                model="llama3",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response["message"]["content"]

        except:
            pass

    # =========================================
    # FALLBACK MULTILINGUAL RESPONSE
    # =========================================

    if lang == "fa":

        return """
توصیه‌های عمومی:

• رژیم غذایی سالم داشته باشید
• مصرف چربی و کلسترول را کاهش دهید
• ورزش منظم انجام دهید
• فشار خون را کنترل کنید
• از سیگار پرهیز کنید
• با پزشک مشورت کنید
"""

    elif lang == "fr":

        return """
Recommandations générales :

• Adoptez une alimentation saine
• Réduisez les graisses et le cholestérol
• Faites de l’exercice régulièrement
• Contrôlez votre tension artérielle
• Évitez de fumer
• Consultez un professionnel de santé
"""

    else:

        return """
General Recommendations:

• Follow a heart-healthy diet
• Reduce saturated fats and cholesterol
• Exercise regularly
• Maintain healthy blood pressure
• Avoid smoking
• Reduce stress
• Consult a healthcare professional
"""

# =========================================
# MULTILINGUAL PROMPT
# =========================================

def build_prompt(profile, risk, lang="en"):

    if lang == "fa":

        return f"""
شما یک دستیار پزشکی هستید.

اطلاعات بیمار:
{profile}

سطح ریسک:
{risk}

توصیه‌های پزشکی ایمن و عمومی ارائه بده.
حتماً توصیه کن با پزشک مشورت شود.
"""

    elif lang == "fr":

        return f"""
Vous êtes un assistant médical.

Profil du patient:
{profile}

Niveau de risque:
{risk}

Fournissez des conseils médicaux généraux et sûrs.
Recommandez de consulter un médecin.
"""

    else:

        return f"""
You are a medical assistant.

Patient Profile:
{profile}

Risk Level:
{risk}

Provide safe and general medical advice.
Always recommend consulting a healthcare professional.
"""

# =========================================
# UI
# =========================================

st.title("❤️ Generative AI Heart Disease Assistant")

st.markdown(
    "Educational & Research Use Only"
)

# =========================================
# USER INPUTS
# =========================================

age = st.slider("Age", 20, 90, 50)

sex = st.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.slider("Chest Pain Type (cp)", 0, 3, 1)

trestbps = st.slider(
    "Resting Blood Pressure",
    80,
    220,
    120
)

chol = st.slider(
    "Cholesterol",
    100,
    400,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

restecg = st.slider(
    "Resting ECG",
    0,
    2,
    1
)

thalach = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.slider(
    "ST Depression (oldpeak)",
    0.0,
    6.0,
    1.0
)

slope = st.slider(
    "Slope",
    0,
    2,
    1
)

ca = st.slider(
    "Number of Major Vessels",
    0,
    4,
    0
)

thal = st.slider(
    "Thal",
    0,
    3,
    2
)

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
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Probability
    risk_prob = model.predict_proba(sample)[0][1]

    # Same threshold as notebook
    prediction = int(risk_prob > 0.3)

    if prediction == 1:
        risk = "HIGH RISK"
        st.error(f"Prediction: {risk}")
    else:
        risk = "LOW RISK"
        st.success(f"Prediction: {risk}")

    st.write(
        f"Risk Probability: {risk_prob:.2f}"
    )

    # Profile
    profile = f"""
Age: {age}
Sex: {sex}
Chest Pain Type: {cp}
Blood Pressure: {trestbps}
Cholesterol: {chol}
Maximum Heart Rate: {thalach}
"""

    # Prompt
    prompt = build_prompt(
        profile,
        risk,
        language
    )

    # AI Response
    response = call_llm(
        prompt,
        language
    )

    st.subheader("AI Medical Advice")

    st.write(response)

# =========================================
# DISCLAIMER
# =========================================

st.warning(
    "This project is intended for educational and research purposes only "
    "and does not provide professional medical advice, diagnosis, or treatment."
)
