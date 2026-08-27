1
import pandas as pd
import numpy as np
import joblib
import shap
import sys
# This is required for SHAP plots, though we aren't displaying them interactively.
import matplotlib.pyplot as plt

# --- 1. Load Trained Artifacts ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("✅ Model and artifacts loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model files not found. Please ensure all .pkl files are in the same directory.")
    sys.exit()

# --- 2. Create SHAP Explainer ---
explainer = shap.TreeExplainer(model)

# --- Helper Functions for User Input ---
def prompt_for_choice(prompt_text, choices):
    """Generic function to prompt user for a choice from a dictionary."""
    print(f"\n{prompt_text}")
    for key, value in choices.items():
        print(f"  {key}) {value}")
    
    while True:
        choice = input("Enter the number for your choice: ")
        if choice in choices:
            return choices[choice]
        print("❌ Invalid input. Please enter a valid number from the list.")

def prompt_for_numeric(prompt_text):
    """Generic function to prompt user for a numeric value."""
    while True:
        try:
            value = float(input(f"\n{prompt_text}: "))
            return value
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

# --- 3. Gather Applicant Data Interactively ---
def get_applicant_input():
    """Asks the user a series of questions to build the applicant data dictionary."""
    print("\n--- Please provide the Applicant's Information ---")
    data = {}

    data['Gender'] = prompt_for_choice("Gender", {'1': 'Male', '2': 'Female'})
    data['Married'] = prompt_for_choice("Marital Status", {'1': 'Yes', '2': 'No'})
    data['Dependents'] = prompt_for_choice("Number of Dependents", {'1': '0', '2': '1', '3': '2', '4': '3+'})
    data['Education'] = prompt_for_choice("Education Level", {'1': 'Graduate', '2': 'Not Graduate'})
    data['Self_Employed'] = prompt_for_choice("Are you Self-Employed?", {'1': 'Yes', '2': 'No'})
    
    credit_history_map = {'1': 1.0, '2': 0.0}
    data['Credit_History'] = prompt_for_choice(
        "Credit History (Have you met credit guidelines in the past?)", 
        {'1': 'Good (Met Guidelines)', '2': 'Bad (Did Not Meet Guidelines)'}
    )
    # Map the text choice back to the numeric value the model expects
    data['Credit_History'] = 1.0 if data['Credit_History'] == 'Good (Met Guidelines)' else 0.0

    data['Property_Area'] = prompt_for_choice("Property Area", {'1': 'Urban', '2': 'Rural', '3': 'Semiurban'})
    
    data['TotalIncome'] = prompt_for_numeric("Enter Total Monthly Income (e.g., 5000)")
    data['LoanAmount'] = prompt_for_numeric("Enter desired Loan Amount (e.g., 150)")
    data['Loan_Amount_Term'] = prompt_for_numeric("Enter Loan Term in Months (e.g., 360)")

    print("\n✅ Thank you. Processing your information...")
    return data

def generate_explanation_summary(explanation, feature_names_list):
    """Generates a plain-language summary of a SHAP explanation."""
    base_value = explanation.base_values
    final_value = np.sum(explanation.values) + base_value
    abs_shap = np.abs(explanation.values)
    max_impact_idx = np.argmax(abs_shap)
    most_impactful_feature = feature_names_list[max_impact_idx]
    impact_value = explanation.values[max_impact_idx]
    feature_input_value = explanation.data[max_impact_idx]

    trend = "an ABOVE-AVERAGE profile" if final_value > base_value else "a BELOW-AVERAGE profile"

    if impact_value > 0.01:
        direction = "significantly INCREASED the chance of approval."
    elif impact_value < -0.01:
        direction = "significantly DECREASED the chance of approval."
    else:
        direction = "had a NEGLIGIBLE impact."

    if most_impactful_feature == 'Credit_History':
        feature_input_value = "Good" if feature_input_value == 1.0 else "Bad"

    summary = (
        f"The model's average prediction is {base_value:.2f}. For this applicant, the final score is {final_value:.2f}, indicating {trend}.\n\n"
        f"The most important factor was '{most_impactful_feature}'. The applicant's value of '{feature_input_value}' {direction}"
    )
    return summary

# --- Main script execution ---
if __name__ == "__main__":
    # Get data from user
    applicant_data = get_applicant_input()
    
    # --- 4. Process Input and Make Prediction ---
    input_df = pd.DataFrame([applicant_data])
    input_df['LoanAmountlog'] = np.log(input_df['LoanAmount'])
    input_df['TotalIncomelog'] = np.log(input_df['TotalIncome'])
    input_df = input_df[feature_names]

    processed_input = input_df.copy()
    for col_idx, encoder in encoders.items():
        col_name = feature_names[col_idx]
        processed_input[col_name] = encoder.transform(processed_input[col_name].values.ravel())

    scaled_input = scaler.transform(processed_input)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    status = target_encoder.inverse_transform(prediction)[0]
    approval_prob = prediction_proba[0][1]

    # --- 5. Display Final Output ---
    print("\n" + "="*40)
    print("--- 🎯 FINAL LOAN ELIGIBILITY ASSESSMENT ---")
    print("="*40)

    if status == 'Y':
        print(f"\nLoan Status: ELIGIBLE ✔️")
    else:
        print(f"\nLoan Status: NOT ELIGIBLE ❌")

    print(f"Eligibility Probability Score: {approval_prob:.1%}\n")
    print("--- 💡 Model Explanation (Why?) ---")
    
    shap_explanation = explainer(scaled_input, check_additivity=False)
    explanation_for_approved = shap_explanation[:,:,1]
    summary = generate_explanation_summary(explanation_for_approved[0], feature_names)
    print(summary)
    print("\n" + "="*40)