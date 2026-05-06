#!/usr/bin/env python
# coding: utf-8

# ## Generative AI for Personalized Heart Disease Education
# This project combines machine learning with retrieval-augmented generation (RAG)
# to generate personalized medical recommendations.

# In[2]:


import pandas as pd
import numpy as np
import requests
import textwrap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

df = pd.read_csv(url, header=None, names=columns, na_values="?")
df.head()


# In[4]:


df.shape


# In[5]:


len(df.columns)


# In[7]:


df.info()


# In[12]:


df = df.dropna()


# In[14]:


# target
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# features
X = df.drop(columns=["num", "target"])
y = df["target"]


# The original UCI Heart Disease dataset was used and stored locally to ensure reproducibility and avoid server availability issues.

# In[17]:


from sklearn.model_selection import train_test_split, cross_val_score

#  split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#  model random forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

#  cross validation 

scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation accuracy:", scores)
print("Mean CV accuracy:", scores.mean())


# In[18]:


print("Std:", scores.std())


# In[19]:


print(X.columns)


# In[20]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# threshold
threshold = 0.3
y_pred_new = (y_prob >= threshold).astype(int)

print("\n=== THRESHOLD TUNING ===")
print("Threshold:", threshold)
print("Accuracy:", accuracy_score(y_test, y_pred_new))
print("Precision:", precision_score(y_test, y_pred_new))
print("Recall:", recall_score(y_test, y_pred_new))
print("F1 Score:", f1_score(y_test, y_pred_new))


# DEFAULT MODEL (threshold=0.5) 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n=== DEFAULT MODEL ===")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")


# In[25]:


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

plt.title("Confusion Matrix")
plt.show()


# In[27]:


ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=["Low Risk", "High Risk"]
)

plt.title("Confusion Matrix")
plt.show()


# In[29]:


def create_profile(row):
    return f"""
Patient is {int(row['age'])} years old.
Cholesterol: {row['chol']}
Blood pressure: {row['trestbps']}
Max heart rate: {row['thalach']}
"""

df["profile_text"] = df.apply(create_profile, axis=1)


# In[31]:


from PyPDF2 import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_text = ""


# In[33]:


def split_text(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = split_text(pdf_text)


# In[35]:


import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection(name="medical_docs")

collection.add(
    documents=chunks,
    ids=[str(i) for i in range(len(chunks))]
)

def retrieve_chunks(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0]


# In[36]:


print(collection.count())


# In[37]:


def build_rag_prompt(profile, query):
    docs = retrieve_chunks(query)
    context = "\n".join(docs)

    return f"""
You are a helpful medical assistant.

Patient profile:
{profile}

Relevant medical guidelines:
{context}

Give personalized advice.
"""


# In[38]:


sample = df.iloc[0]

profile = create_profile(sample)


# In[39]:


print(X.columns)


# In[40]:


X_sample = sample[X.columns]


# In[41]:


X_sample = pd.DataFrame([sample[X.columns]])
prediction = model.predict(X_sample)[0]


# In[49]:


print(X_sample.shape)


# In[51]:


if prediction == 1:
    risk = "HIGH risk"
else:
    risk = "LOW risk"


# In[58]:


import ollama

def call_llm(prompt, system_prompt):
    response = ollama.chat(
        model='llama3',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content']

SYSTEM_PROMPT_3 = "You are a medical assistant."


# In[60]:


prompt = build_rag_prompt(profile + "\nRisk: " + risk, "how to reduce cholesterol")


# In[62]:


response = call_llm(prompt, SYSTEM_PROMPT_3)


# In[63]:


print(response)


# In[66]:


import sys
get_ipython().system('{sys.executable} -m pip install PyPDF2')


# In[67]:


pdf_path = "How-Can-I-Improve-Cholesterol.pdf"


# In[70]:


from PyPDF2 import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# In[72]:


pdf_text = ""


# In[74]:


def retrieve_chroma(query, top_k=2):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0]


# In[76]:


docs = retrieve_chroma("how to reduce cholesterol")


# In[78]:


def build_rag_prompt(profile, query):
    docs = retrieve_chroma(query)
    context = "\n".join(docs)

    prompt = f"""
You are a helpful medical assistant.

Patient profile:
{profile}

Relevant medical guidelines:
{context}

Based on the above, give personalized advice.
"""
    return prompt


# In[80]:


SYSTEM_PROMPT = """
You are a medical assistant.

ONLY answer medical-related questions.
If the user asks anything unrelated (movies, politics, etc),
respond with:
"Sorry, I can only answer medical-related questions."
"""


# In[82]:


SYSTEM_PROMPT_3 = "You are a helpful medical assistant. Give clear and safe advice."


# In[84]:


get_ipython().system('pip install ollama')


# In[86]:


import ollama
print("OK")


# In[88]:


def call_llm(prompt, system_prompt):
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


# In[90]:


# sample input
sample = df.iloc[0]

# create profile
profile = create_profile(sample)

# prepare ML input
X_sample = sample[X.columns]
X_sample = pd.DataFrame([X_sample])

# prediction
prediction = model.predict(X_sample)[0]

# risk label
if prediction == 1:
    risk = "HIGH risk"
else:
    risk = "LOW risk"

# build RAG prompt
prompt = build_rag_prompt(profile + "\nRisk: " + risk, "how to reduce cholesterol")


response = call_llm(prompt, SYSTEM_PROMPT_3)

# output
print("=== PATIENT PROFILE ===")
print(profile)

print("\n=== RISK ===")
print(risk)

print("\n=== AI RESPONSE ===")
print(response)


# In[91]:


sample = df.iloc[0]

#
profile = create_profile(sample)

# feature train
X_sample = sample[X.columns]

#  DataFrame
X_sample = pd.DataFrame([X_sample])

# probability
risk_prob = model.predict_proba(X_sample)[0][1]

# threshold 
prediction = int(risk_prob > 0.3)

print(f"Risk Probability: {risk_prob:.2f}")
print("Prediction:", prediction)
print(profile)


# In[92]:


if prediction == 1:
    risk = "HIGH risk"
else:
    risk = "LOW risk"

prompt = build_rag_prompt(profile + "\n" + risk, "how to reduce cholesterol")

response = call_llm(prompt, SYSTEM_PROMPT_3)
print(response)


# In[93]:


def build_rag_prompt(profile, docs):
    context = " ".join(docs)

    return f"""
Patient Profile:
{profile}

Medical Guidelines:
{context}

Give personalized medical advice.
"""


# In[94]:


SYSTEM_PROMPT_1 = "You are a helpful medical assistant."

SYSTEM_PROMPT_2 = """
You are a professional medical doctor.
Give clear, structured, and accurate medical advice.
Use bullet points.
"""

SYSTEM_PROMPT_3 = """
You are a medical assistant.

Only answer medical-related questions.
If the question is not medical, say:
"I can only answer medical-related questions."
"""


# In[95]:


print("=== PROMPT COMPARISON ===\n")

print("---- Prompt 1 ----")
print(call_llm(prompt, SYSTEM_PROMPT_1))

print("\n---- Prompt 2 ----")
print(call_llm(prompt, SYSTEM_PROMPT_2))

print("\n---- Prompt 3 ----")
print(call_llm(prompt, SYSTEM_PROMPT_3))


# In[104]:


def call_llm(prompt, system_prompt):
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


# In[106]:


print(call_llm("How to reduce cholesterol?", SYSTEM_PROMPT_1))


# In[107]:


print(call_llm("How to reduce cholesterol?", SYSTEM_PROMPT_2))


# In[108]:


print(call_llm("Tell me about movies", SYSTEM_PROMPT_3))


# In[110]:


def build_rag_prompt(profile, docs):
    context = "\n".join(docs)

    prompt = f"""
Patient Information:
{profile}

Medical Guidelines:
{context}

Based on the above, give personalized medical advice.
"""
    return prompt


# In[114]:


rag_outputs = []
baseline_outputs = []


# In[116]:


baseline_prompt = f"""
You are a medical assistant.

Patient Profile:
{profile}

Give advice.
"""

baseline_response = call_llm(baseline_prompt, SYSTEM_PROMPT_3)
baseline_outputs.append(baseline_response)


# In[117]:


rag_prompt = build_rag_prompt(profile, "how to reduce cholesterol")
rag_response = call_llm(rag_prompt, SYSTEM_PROMPT_3)
rag_outputs.append(rag_response)


# In[118]:


print("=== BASELINE ===")
print(baseline_outputs[0])

print("\n=== RAG ===")
print(rag_outputs[0])


# In[119]:


#  FINAL RAG + ML + LLM PIPELINE 
sample = df.iloc[0]

#  Profile
sample = df.iloc[0]

# Profile
profile = sample["profile_text"]

#  ML prediction 
X_sample = sample[X.columns]
X_sample = pd.DataFrame([X_sample])

risk_prob = model.predict_proba(X_sample)[0][1]

print(f"Predicted Risk Probability: {risk_prob:.2f}")


# In[120]:


#basline


baseline_prompt = f"""

You are a medical assistant.

Patient Profile:

{profile}

Predicted risk probability: {risk_prob:.2f}

Provide general lifestyle advice for heart health.

"""

baseline_response = call_llm(baseline_prompt, SYSTEM_PROMPT_1)


# In[121]:


print(baseline_response)


# In[128]:


#rag
query = "how to reduce cholesterol and heart disease risk"
docs = retrieve_chroma(query)

prompt = build_rag_prompt(
    profile + f"\nPredicted risk probability: {risk_prob:.2f}",
    docs
)

rag_response = call_llm(prompt, SYSTEM_PROMPT_2)

print("\n=== RAG OUTPUT ===\n")
print(rag_response)


# In[129]:


context = "\n".join(docs)


# In[130]:


restricted_prompt = f"""
You are a medical assistant.

ONLY answer medical-related questions.
If the question is not medical, say:
"Sorry, I can only answer medical-related questions."

Patient profile:
{profile}

Medical guidelines:
{context}

Predicted risk probability: {risk_prob:.2f}

Give personalized medical advice.
"""


# In[131]:


restricted_response = call_llm(restricted_prompt, SYSTEM_PROMPT_3)

print("\n=== RESTRICTED OUTPUT ===\n")

print(restricted_response)


# In[132]:


baseline_outputs.append(baseline_response)
rag_outputs.append(rag_response)


# In[133]:


len(baseline_outputs), len(rag_outputs)


# In[134]:


def is_hallucination(output):
    text = output.lower()
    
    score = 0
    
    if "statin" in text or "ace inhibitor" in text:
        score += 1
        
    if "you must take" in text or "you should take" in text:
        score += 1
        
    if "consult" not in text:
        score += 1
        
    return score >= 2

baseline_h = sum(is_hallucination(o) for o in baseline_outputs)
rag_h = sum(is_hallucination(o) for o in rag_outputs)

print("Baseline Hallucinations:", baseline_h)
print("RAG Hallucinations:", rag_h)


# In[135]:


print("Baseline Rate:", baseline_h / len(baseline_outputs))
print("RAG Rate:", rag_h / len(rag_outputs))


# In[136]:


print("=== COMPARISON: BASELINE vs RAG ===")

print("\n=== BASELINE OUTPUT ===")
print(baseline_response)

print("\n=== RAG OUTPUT ===")
print(rag_response)


# In[137]:


print(sample.keys())


# In[138]:


sample = df.iloc[0]

# Profile
profile = sample["profile_text"]

# ML prediction 

X_sample = sample[X.columns]   # 
X_sample = pd.DataFrame([X_sample])

risk_prob = model.predict_proba(X_sample)[0][1]


print(f"Predicted Risk Probability: {risk_prob:.2f}")

# Retrieval
query = f"how to reduce cholesterol and heart disease risk"
docs = retrieve_chroma(query)

# Build prompt
prompt = build_rag_prompt(
    profile + f"\nPredicted risk probability: {risk_prob:.2f}",
    docs
)

#  LLM  
response = call_llm(prompt, SYSTEM_PROMPT_3)

print("\n=== FINAL RESPONSE ===")
print(response)


# ## Final System
# 
# This project integrates:
# - Machine Learning for risk prediction
# - Retrieval-Augmented Generation (RAG) using ChromaDB
# - Large Language Model (LLM) for personalized medical advice
# 
# The system generates grounded and personalized recommendations based on both patient data and medical guidelines.

# ## Final Comparison
# 
# The baseline model provides general advice without using external knowledge.
# 
# In contrast, the RAG-based system retrieves relevant medical guidelines from external sources using ChromaDB and generates more accurate and personalized recommendations.
# 
# This demonstrates that combining Machine Learning predictions with Retrieval-Augmented Generation significantly improves the quality and reliability of medical advice.

# Overall, the RAG-based system produces more context-aware, reliable, and clinically relevant recommendations compared to the baseline approach.

# In[142]:


get_ipython().system('pip install click==8.1.7 typer==0.9.0')


# In[143]:


import gradio as gr
print("Gradio OK")


# In[144]:


import gradio as gr

def medical_chat(user_input):
    return call_llm(user_input, SYSTEM_PROMPT_3)

interface = gr.Interface(
    fn=medical_chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask a medical question..."),
    outputs="text",
    title="Medical AI Assistant 🩺",
    description="Only answers medical questions."
)

interface.launch() 


# ### This project is intended for educational and research purposes only and does not provide professional medical advice, diagnosis, or treatment

# In[172]:


import sys
get_ipython().system('{sys.executable} -m pip install langdetect')


# In[174]:


from langdetect import detect


# In[176]:


from langdetect import detect

def get_language(text):
    try:
        return detect(text)
    except:
        return "en"


# In[178]:


def build_multilingual_prompt(profile, risk_prob, context, lang="en"):
    
    if lang == "fa":  # Persian
        return f"""
شما یک دستیار پزشکی هستید.

اطلاعات بیمار:
{profile}

احتمال ریسک پیش‌بینی‌شده: {risk_prob:.2f}

راهنماهای پزشکی:
{context}

بر اساس اطلاعات بالا، توصیه‌های پزشکی شخصی‌سازی‌شده ارائه بده.
حتماً توصیه کن که بیمار با پزشک مشورت کند.
"""

    elif lang == "fr":  # French
        return f"""
Vous êtes un assistant médical.

Profil du patient:
{profile}

Probabilité de risque prédite: {risk_prob:.2f}

Directives médicales:
{context}

Sur la base des informations ci-dessus, fournissez des conseils médicaux personnalisés.
Assurez-vous de recommander de consulter un médecin.
"""

    else:  # English
        return f"""
You are a medical assistant.

Patient Profile:
{profile}

Predicted Risk Probability: {risk_prob:.2f}

Medical Guidelines:
{context}

Based on the above, provide personalized medical advice.
Always recommend consulting a healthcare professional.
"""


# In[180]:


def get_query_by_language(lang):
    if lang == "fa":
        return "چگونه کلسترول را کاهش دهیم و خطر بیماری قلبی را کم کنیم"
    elif lang == "fr":
        return "Comment réduire le cholestérol et le risque de maladie cardiaque"
    else:
        return "How to reduce cholesterol and heart disease risk"


# In[182]:


def is_hallucination(output):
    text = output.lower()
    score = 0

    # risky medication suggestions
    if any(word in text for word in ["statin", "ace inhibitor"]):
        score += 1

    # strong forcing language
    if any(word in text for word in [
        "you must take", "you should take",
        "vous devez", "devez prendre",
        "باید مصرف کنید", "حتما مصرف کنید"
    ]):
        score += 1

    # missing doctor consultation warning
    if not any(word in text for word in [
        "consult", "consultez", "پزشک"
    ]):
        score += 1

    return score >= 2


# In[184]:


def multilingual_pipeline(user_input, profile, risk_prob):

    # 1. detect language
    lang = get_language(user_input)

    # 2. get query
    query = get_query_by_language(lang)

    # 3. retrieve documents
    docs = retrieve_chroma(query)   # همون تابع قبلی خودت

    context = "\n".join(docs)

    # 4. build prompt
    prompt = build_multilingual_prompt(profile, risk_prob, context, lang)

    # 5. call LLM
    response = call_llm(prompt, SYSTEM_PROMPT_3)

    return response


# In[186]:


test_inputs = [
    "How can I reduce cholesterol?",
    "چطور کلسترولم را کاهش دهم؟",
    "Comment réduire mon cholestérol ?"
]

for text in test_inputs:
    response = multilingual_pipeline(text, profile, risk_prob)
    
    print("\n==============================")
    print("Input:", text)
    print("Response:", response)
    print("Hallucination:", is_hallucination(response))


# In[188]:


from langdetect import detect

def get_language(text):
    try:
        return detect(text)
    except:
        return "en"


# In[190]:


def get_system_prompt(lang):
    if lang == "fa":
        return """You are a medical assistant.
Respond in Persian.
Give safe, helpful, and evidence-based advice."""
    
    elif lang == "fr":
        return """You are a medical assistant.
Respond in French.
Give safe, helpful, and evidence-based advice."""
    
    elif lang == "ur":
        return """You are a medical assistant.
Respond in Urdu.
Give safe, helpful, and evidence-based advice."""
    
    else:
        return """You are a medical assistant.
Respond in English.
Give safe, helpful, and evidence-based advice."""


# In[192]:


def multilingual_pipeline(text, profile, risk_prob):
    
    # 1. detect language
    lang = get_language(text)
    
    # 2. build prompt (RAG)
    prompt = build_rag_prompt(
        profile + f"\nRisk probability: {risk_prob:.2f}",
        text
    )
    
    # 3. dynamic system prompt
    system_prompt = get_system_prompt(lang)
    
    # 4. call LLM
    response = call_llm(prompt, system_prompt)
    
    return response


# In[194]:


test_inputs = [
    "How can I reduce cholesterol?",
    "چطور کلسترولم را کاهش دهم؟",
    "Comment réduire mon cholestérol ?",
    "میں اپنا کولیسٹرول کیسے کم کروں؟"
]


# In[197]:


for text in test_inputs:
    response = multilingual_pipeline(text, profile, risk_prob)
    
    print("\n============================")
    print("Input:", text)
    print("Response:", response)


# In[ ]:




