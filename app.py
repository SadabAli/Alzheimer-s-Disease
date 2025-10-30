# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from dotenv import load_dotenv
import google.generativeai as genai

# Optional: for nicer feature importance visualization
import matplotlib.pyplot as plt

# ------------------------
# Load environment
# ------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("MODEL_NAME", "models/gemini-2.5-pro")

if API_KEY is None:
    st.error("No GOOGLE_API_KEY found in .env. Add your Gemini API key to .env file.")
    st.stop()

# Configure Gemini
genai.configure(api_key=API_KEY)

# ------------------------
# Load saved CatBoost model
# ------------------------
MODEL_PATH = "alzheimers_catboost_model.cbm"  # change path if required

@st.cache_resource(show_spinner=False)
def load_model(path):
    m = CatBoostClassifier()
    m.load_model(path)
    return m

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load CatBoost model from '{MODEL_PATH}': {e}")
    st.stop()

# ------------------------
# UI: Title and explanation
# ------------------------
st.title("Alzheimer's Risk Predictor — AI Assisted")
st.markdown(
    """
Enter the patient's clinical values (the 18 selected features are used).
The system will predict Alzheimer’s risk (0 = No, 1 = Yes) using the trained CatBoost model.
After prediction, a Gemini-generated explanation and suggestions will be shown.
"""
)

# ------------------------
# Input fields (17 features)
# Note: adjust ranges/choices depending on your data encoding
# You listed these features:
# ['Age','BMI','PhysicalActivity','DietQuality','SleepQuality','HeadInjury',
#  'Hypertension','CholesterolTotal','CholesterolLDL','CholesterolHDL',
#  'CholesterolTriglycerides','MMSE','FunctionalAssessment','MemoryComplaints',
#  'BehavioralProblems','ADL','PersonalityChanges','Diagnosis']
# Diagnosis is target so we do inputs for others (17 inputs)
# ------------------------

with st.form("input_form"):
    st.subheader("Patient data")
    age = st.number_input("Age", min_value=18, max_value=120, value=70)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
    physical_activity = st.selectbox("Physical Activity (0=low,1=moderate,2=high)", [0,1,2], index=0)
    diet_quality = st.selectbox("Diet Quality (0=poor,1=average,2=good)", [0,1,2], index=1)
    sleep_quality = st.selectbox("Sleep Quality (0=poor,1=average,2=good)", [0,1,2], index=1)
    head_injury = st.selectbox("Head Injury (0=no,1=yes)", [0,1], index=0)
    hypertension = st.selectbox("Hypertension (0=no,1=yes)", [0,1], index=0)
    chol_total = st.number_input("Cholesterol Total (mg/dL)", min_value=50.0, max_value=400.0, value=200.0, step=1.0)
    chol_ldl = st.number_input("Cholesterol LDL (mg/dL)", min_value=10.0, max_value=300.0, value=120.0, step=1.0)
    chol_hdl = st.number_input("Cholesterol HDL (mg/dL)", min_value=5.0, max_value=120.0, value=45.0, step=1.0)
    chol_tri = st.number_input("Triglycerides (mg/dL)", min_value=20.0, max_value=500.0, value=150.0, step=1.0)
    mmse = st.number_input("MMSE (Mini-Mental State Exam) Score", min_value=0, max_value=30, value=25)
    func_assess = st.number_input("Functional Assessment (0-10)", min_value=0, max_value=30, value=5)
    memory_complaints = st.selectbox("Memory Complaints (0=no,1=yes)", [0,1], index=1)
    behavioral = st.selectbox("Behavioral Problems (0=no,1=yes)", [0,1], index=0)
    adl = st.number_input("ADL (Activities of Daily Living) score (0-20)", min_value=0, max_value=30, value=15)
    personality_changes = st.selectbox("Personality Changes (0=no,1=yes)", [0,1], index=0)

    submitted = st.form_submit_button("Predict & Generate Suggestion")

# ------------------------
# Prepare input for model
# ------------------------
def build_input_df():
    df = pd.DataFrame([{
        'Age': age,
        'BMI': bmi,
        'PhysicalActivity': physical_activity,
        'DietQuality': diet_quality,
        'SleepQuality': sleep_quality,
        'HeadInjury': head_injury,
        'Hypertension': hypertension,
        'CholesterolTotal': chol_total,
        'CholesterolLDL': chol_ldl,
        'CholesterolHDL': chol_hdl,
        'CholesterolTriglycerides': chol_tri,
        'MMSE': mmse,
        'FunctionalAssessment': func_assess,
        'MemoryComplaints': memory_complaints,
        'BehavioralProblems': behavioral,
        'ADL': adl,
        'PersonalityChanges': personality_changes
    }])
    return df

# ------------------------
# Prediction + Gemini interaction
# ------------------------
if submitted:
    input_df = build_input_df()

    # Model prediction
    try:
        prob = model.predict_proba(input_df)[0][1]  # probability of class 1
        pred = int(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    st.markdown("## Prediction Result")
    st.write(f"**Predicted class:** {pred}  &nbsp;&nbsp;  **Probability of positive (AD)**: {prob:.3f}")

    # Feature importances (if available)
    try:
        fi = model.get_feature_importance(pool=None)  # CatBoost returns array matching feature order
        feature_names = input_df.columns.tolist()
        fi_series = pd.Series(fi, index=feature_names).sort_values(ascending=False)
        st.markdown("**Top features by CatBoost importance**")
        st.table(fi_series.head(8).reset_index().rename(columns={'index':'feature', 0:'importance'}))
        # simple bar plot
        fig, ax = plt.subplots(figsize=(6,3))
        fi_series.head(8).plot(kind='barh', ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    except Exception:
        # fallback: show no importance
        st.info("Feature importance not available for this model instance.")

    # Prepare prompt for Gemini
    # Keep prompt clear and use bullet points: user input + model result + request for actionable suggestions
    prompt_lines = []
    prompt_lines.append("You are a clinical assistant helping a doctor interpret a machine learning prediction for Alzheimer's disease.")
    prompt_lines.append("Patient data (numeric/binary fields):")
    for k, v in input_df.iloc[0].items():
        prompt_lines.append(f"- {k}: {v}")
    prompt_lines.append(f"Model prediction: class {pred} (1 = Alzheimer's likely, 0 = not likely), probability = {prob:.3f}.")
    prompt_lines.append("")
    prompt_lines.append("Please provide:")
    prompt_lines.append("1) a short (2-3 sentence) summary of why the model predicted this (use medical reasoning),")
    prompt_lines.append("2) three practical suggestions / next steps (one sentence each) for clinicians or patient care, and")
    prompt_lines.append("3) a simple plain-language paragraph (2-3 sentences) the patient can understand.")
    prompt = "\n".join(prompt_lines)

    # Call Gemini to generate suggestions
    st.markdown("### Generating AI suggestions (Gemini)...")
    try:
        model_api = genai.GenerativeModel(GEMINI_MODEL)
        # Some SDK variants use .generate() or .generate_content(); both may work depending on SDK version.
        # We'll try generate_content first (common in examples).
        resp = model_api.generate_content(prompt)
        # response text may be at resp.text or resp.generations[0].text depending on SDK
        gen_text = ""
        if hasattr(resp, "text"):
            gen_text = resp.text
        elif hasattr(resp, "generations"):
            # older/newer responses
            gen_text = "\n".join([g.text for g in resp.generations])
        else:
            gen_text = str(resp)

        st.markdown("#### AI Explanation & Suggestions")
        st.write(gen_text)

    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        st.info("You can still view the model prediction above. Ensure your API key and model name are correct.")


# Footer
st.markdown("---")
st.caption("Alzheimer's predictor demo — Model trained with CatBoost. Gemini provides textual explanations. Not a medical device; for research/demo only.")
