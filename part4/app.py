import os
import re
import json
import joblib
import requests
import pandas as pd
import jsonschema
from jsonschema import validate
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("LLM_API_KEY")
API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

### task1:
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    if not API_KEY:
        print("Error: LLM_API_KEY not found in environment variables.")
        return None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model":MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}

        ],
        "temperature": temperature,
        "max_tokens": max_tokens

    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code !=200:
            print(f"API Error Response [Status {response.status_code}]: {response.text}")
            return None

        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"HTTP Request Exception: {e}")
        return None


test_system = "You are a helpful assistant."
test_user = "Reply with only the word: hello"
test_response = call_llm(test_system, test_user, temperature=0.0)
print(f"Test Prompt Response: '{test_response}'")

### before start the next task we need to get set guardrail 
### task4:
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
pii_input_dirty = "Customer john.doe@example.com with tenure 12 has churned."
pii_input_clean = "Customer with tenure 12, monthly charges 85.50, and Fiber Optic service."
for sample_text in [pii_input_dirty, pii_input_clean]:
    if has_pii(sample_text):
        print(f"Input: '{sample_text}' --> Input blocked: PII detected.")
    else:
        print(f"Input: '{sample_text}' --> Guardrail Passed. Proceeding to LLM call.")


### task3:
model_path = "best_model.pkl"
if not os.path.exists(model_path):
    model_path = "../best_model.pkl" if os.path.exists("../best_model.pkl") else "best_model.pkl"
best_pipeline = joblib.load(model_path)
print(f"Loaded pipeline model successfully from '{model_path}'")
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}

    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"
    ]
}
SYSTEM_PROMPT = (
    "You are an AI explanation assistant for a customer churn machine learning pipeline. "
    "Your task is to analyze feature-vector inputs, model predicted classes, and probabilities, "
    "and produce a structured JSON response explaining the prediction. "
    "Output strictly valid JSON with no conversational preamble or markdown code blocks outside of raw JSON."

)
USER_PROMPT_TEMPLATE = """
Evaluate the following model prediction and return a JSON object following the exact schema required.

Feature Values: {feature_values}
Predicted Class: {predicted_class} (0 = Retained, 1 = Churned)
Predicted Churn Probability: {predicted_prob:.4f}

Required Output JSON Schema Keys:
- "prediction_label": String description ("Likely Churn" or "Likely Retained")
- "confidence_level": String ("low", "medium", or "high")
- "top_reason": String identifying the strongest factor contributing to this prediction
- "second_reason": String identifying the second strongest contributing factor
- "next_step": String offering an actionable business retention or maintenance step
"""
test_features_list = [
    {
        "gender": "Female", "SeniorCitizen": 0, "tenure": 2.0, "PhoneService": "Yes",
        "InternetService": "Fiber optic", "Contract": "Month-to-Month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "TotalCharges": 150.0

    },
    {
        "gender": "Male", "SeniorCitizen": 0, "tenure": 60.0, "PhoneService": "Yes",
        "InternetService": "Fiber optic", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Credict card", "TotalCharges": 3200.0

    },
    {
        "gender": "Female", "SeniorCitizen": 1, "tenure": 24.0, "PhoneService": "Yes",
        "InternetService": "Fiber optic", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check", "TotalCharges": 1800.0
    }
]

def process_track_c(input_features, temp=0.0):
    input_str = json.dumps(input_features)
    if has_pii(input_str):
        print("Input blocked: PII detected.")
        return None, "Blocked (PII Dectected)"
    
    input_df = pd.DataFrame([input_features])
    input_df_encoded = pd.get_dummies(input_df)
    expected_columns = best_pipeline.feature_names_in_ if hasattr(best_pipeline, 'feature_names_in_') else best_pipeline.steps[-1][1].feature_names_in_
    for col in expected_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_final = input_df_encoded[expected_columns]       
    pred_class = int(best_pipeline.predict(input_df_final)[0])
    pred_prob = float(best_pipeline.predict_proba(input_df_final)[0][1])
    formatted_user_prompt = USER_PROMPT_TEMPLATE.format(
        feature_values=input_str,
        predicted_class=pred_class,
        predicted_prob=pred_prob

    )
    raw_response = call_llm(SYSTEM_PROMPT, formatted_user_prompt, temperature=temp)
    if not raw_response:
        fallback = {
            "prediction_label": None, "confidence_level": None,
            "top_reason": None, "second_reason": None, "next_step": None
        }
        return fallback, "Fail (API Error)"
    clean_text = raw_response.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        parsed_json = json.loads(clean_text)
        validate(instance=parsed_json, schema=EXPLANATION_SCHEMA)
        return parsed_json, "Pass"
    except (json.JSONDecodeError, jsonschema.ValidationError) as err:
        print(f"Validation Error encountered: {err}")
        fallback = {
            "prediction_label": None, "confidence_level": None,
            "top_reason": None, "second_reason": None, "next_step": None

        }
        return fallback, f"Fail ({type(err).__name__})"
    
### task2:
for idx, features in enumerate(test_features_list, 1):
    print(f"\nEvaluating Input case #{idx}:")
    out_t0, status_t0 = process_track_c(features, temp=0.0)
    out_t07, status_t07 = process_track_c(features, temp=0.7)
    print(f"Output @ temp=0.0 (Status: {status_t0}):\n{json.dumps(out_t0, indent=2)}")
    print(f"Output @ temp=0.7 (Status: {status_t07}):\n{json.dumps(out_t07, indent=2)}")

### task5:
demo_summary = []
for idx, features in enumerate(test_features_list, 1):
    print(f"Raw Input Data:")
    print(features)
    print()
    explanation_json, val_status = process_track_c(features, temp=0.0)
    ai_label = explanation_json.get("prediction_label", "N/A")
    ai_confidence = explanation_json.get("confidence_level", "N/A")
    input_df = pd.DataFrame([features])
    input_df_encoded = pd.get_dummies(input_df)
    expected_columns = (
        best_pipeline.feature_names_in_ 
        if hasattr(best_pipeline, 'feature_names_in_') 
        else best_pipeline.steps[-1].feature_names_in_
    )
    for col in expected_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_final = input_df_encoded[expected_columns]
    
    pred_class = int(best_pipeline.predict(input_df_final)[0])
    pred_prob = float(best_pipeline.predict_proba(input_df_final)[0][1])
    demo_summary.append({
        "Input ID": f"Case #{idx}",
        "Model Guess": "Retained (0)" if pred_class == 0 else "Churn (1)",
        "Risk Prob": f"{pred_prob * 100:.1f}%",
        "AI Verdict": ai_label,
        "AI Conf": ai_confidence,
        "Valid JSON": "pass" if val_status == "Pass" else "fail",
        "Guardrail": "Cleared"
    })
print("FINAL END-TO-END DEMONSTRATION SUMMARY SCOREBOARD")
summary_df = pd.DataFrame(demo_summary)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 22)
print(summary_df.to_string(index=False, justify='left'))

