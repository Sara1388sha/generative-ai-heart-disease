# ============================================
# Generative AI for Heart Disease Education
# Educational & Research Use Only
# ============================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================================
# SAFE OLLAMA IMPORT (Cloud-safe)
# ============================================
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


SYSTEM_PROMPT = """
You are a medical assistant.
Give safe, general, evidence-based advice.
Always recommend consulting a healthcare professional.
"""


# ============================================
# LLM FUNCTION (NO INDENTATION ISSUES)
# ============================================
def call_llm(prompt, system_prompt):
    if not OLLAMA_AVAILABLE:
        return (
            "LLM is disabled in this environment.\n"
            "Run locally with Ollama installed to enable AI responses."
        )

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


# ============================================
# LOAD DATA
# ============================================
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


# ============================================
# TRAIN MODEL
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# ============================================
# HELPER
# ============================================
def create_profile(row):
    return f"""
Age: {int(row['age'])}
Cholesterol: {row['chol']}
Blood Pressure: {row['trestbps']}
Max Heart Rate: {row['thalach']}
"""

# ============================================
# MAIN PIPELINE
# ============================================
if __name__ == "__main__":

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

    print("=== PATIENT PROFILE ===")
    print(profile)

    print(f"Predicted Risk Probability: {risk_prob:.2f}")
    print("Risk Level:", risk_level)

    print("\n=== AI RESPONSE ===")
    print(response)
