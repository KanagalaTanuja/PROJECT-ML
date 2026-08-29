import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap

# Set page layout
st.set_page_config(layout="wide")
st.title("💡 REAL TIME LOAN ELIGIBILITY ASSESSMENT")
st.write("Using Machine Learning and XAI Techniques")

# Train a synthetic model to provide realistic SHAP explanations
@st.cache_resource
def load_model_and_shap():
    # 10 features matching standard loan data
    np.random.seed(42)
    N = 1000
    X = pd.DataFrame({
        'Gender': np.random.choice([0, 1], size=N),
        'Married': np.random.choice([0, 1], size=N),
        'Dependents': np.random.choice([0, 1, 2, 3], size=N),
        'Education': np.random.choice([0, 1], size=N),
        'Self_Employed': np.random.choice([0, 1], size=N),
        'Credit_History': np.random.choice([0, 1], p=[0.2, 0.8], size=N),
        'Property_Area': np.random.choice([0, 1, 2], size=N),
        'Loan_Amount': np.random.uniform(10, 700, size=N),
        'Total_Income': np.random.uniform(1500, 81000, size=N),
        'Loan_Amount_Term': np.random.choice([120, 180, 240, 360, 480], size=N)
    })
    
    # Target rule heavily weighted on Credit History and Income/Loan ratio
    y = (
        (X['Credit_History'] == 1) & 
        ((X['Total_Income'] / (X['Loan_Amount'] + 1)) > 10)
    ).astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns

model, explainer, feature_names = load_model_and_shap()

# Sidebar inputs
st.sidebar.header("Applicant Information")

gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
married = st.sidebar.selectbox('Married', ('Yes', 'No'))
dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))
education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
self_employed = st.sidebar.selectbox('Self Employed', ('Yes', 'No'))
credit_history = st.sidebar.selectbox('Credit History', (1.0, 0.0), format_func=lambda x: 'Good' if x == 1.0 else 'Bad')
property_area = st.sidebar.selectbox('Property Area', ('Urban', 'Rural', 'Semiurban'))
loan_amount = st.sidebar.slider('Loan Amount ($)', 10, 700, 150)
total_income = st.sidebar.slider('Total Monthly Income ($)', 1500, 81000, 5000)
loan_amount_term = st.sidebar.slider('Loan Amount Term (Months)', 36, 480, 360)

# Map user input into numerical model input
dep_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
prop_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

user_features = pd.DataFrame([{
    'Gender': 1 if gender == 'Male' else 0,
    'Married': 1 if married == 'Yes' else 0,
    'Dependents': dep_map[dependents],
    'Education': 1 if education == 'Graduate' else 0,
    'Self_Employed': 1 if self_employed == 'Yes' else 0,
    'Credit_History': int(credit_history),
    'Property_Area': prop_map[property_area],
    'Loan_Amount': loan_amount,
    'Total_Income': total_income,
    'Loan_Amount_Term': loan_amount_term
}])

# Display input summary table
st.subheader("Applicant's Input")
input_display = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'Credit_History': 'Good' if credit_history == 1.0 else 'Bad',
    'Property_Area': property_area,
    'Loan Amount': f"${loan_amount}",
    'Total Income': f"${total_income}",
    'Loan Term': f"{loan_amount_term} months"
}
st.table(pd.DataFrame([input_display]))

if st.sidebar.button("Predict Loan Status", type="primary"):
    # Calculate Prediction Probabilities & SHAP values
    prob = model.predict_proba(user_features)[0][1]
    shap_explanation = explainer(user_features)
    
    # Extract binary classification SHAP (positive outcome class)
    if len(shap_explanation.shape) == 3:
        shap_values_obj = shap_explanation[0, :, 1]
    else:
        shap_values_obj = shap_explanation[0]

    st.subheader("📊 Prediction Result")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if prob >= 0.5:
            st.metric(label="Loan Status", value="✅ APPROVED", delta=f"Score: {prob*100:.0f}/100")
            st.balloons()
        else:
            st.metric(label="Loan Status", value="❌ REJECTED", delta=f"Score: {prob*100:.0f}/100", delta_color="inverse")
            st.toast('The application did not meet the required criteria.', icon='😞')

    with col2:
        st.write("Eligibility Probability Score:")
        st.progress(prob)
        st.markdown(f"**{prob:.1%}**")

    st.markdown("---")
    st.subheader("Why did the model decide this? (XAI Explanation)")

    base_val = shap_values_obj.base_values
    top_feature_idx = np.argmax(np.abs(shap_values_obj.values))
    top_feature_name = feature_names[top_feature_idx]
    
    st.write(f"The model's average prediction (base value) is **{base_val:.2f}**. For this applicant, the final score is **{prob:.2f}**.")
    st.write(f"The most impactful factor was **{top_feature_name}**.")

    # -------------------------------------------------------------
    # Side-by-Side Plots (Waterfall Plot & Feature Impact Bar Plot)
    # -------------------------------------------------------------
    plot_col1, plot_col2 = st.columns(2)

    # Left Column: Waterfall Plot
    with plot_col1:
        st.markdown("#### Waterfall Plot")
        fig_waterfall, ax1 = plt.subplots(figsize=(6, 5))
        shap.plots.waterfall(shap_values_obj, show=False)
        plt.tight_layout()
        st.pyplot(fig_waterfall)
        plt.clf()

    # Right Column: Feature Impact Bar Plot
    with plot_col2:
        st.markdown("#### Feature Impact Bar Plot")
        fig_bar, ax2 = plt.subplots(figsize=(6, 5))
        
        vals = shap_values_obj.values
        sorted_idx = np.argsort(np.abs(vals))
        
        y_pos = np.arange(len(vals))
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_vals = vals[sorted_idx]
        
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in sorted_vals]
        
        ax2.barh(y_pos, sorted_vals, color=colors, height=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_names)
        ax2.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('SHAP value')
        
        for idx, val in enumerate(sorted_vals):
            offset = 0.005 if val >= 0 else -0.005
            ha = 'left' if val >= 0 else 'right'
            ax2.text(val + offset, idx, f"{val:+.2f}", va='center', ha=ha, fontsize=8, fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.clf()
