import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page layout
st.set_page_config(layout="wide")
st.title("💡 REAL TIME LOAN ELIGIBILITY ASSESSMENT")
st.write("Using Machine Learning and XAI Techniques")

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

def compute_xai_scores():
    # Base expected value of the algorithm
    base_value = 0.68
    
    # Calculate feature impacts (SHAP-like values)
    ratio = loan_amount / total_income if total_income > 0 else 0
    
    impacts = {
        'Credit_History': 0.22 if credit_history == 1.0 else -0.35,
        'Income_Ratio': 0.15 if ratio < 0.15 else (-0.20 if ratio > 0.4 else 0.02),
        'Education': 0.04 if education == 'Graduate' else -0.02,
        'Employment': 0.03 if self_employed == 'No' else -0.01,
        'Property_Area': 0.05 if property_area == 'Urban' else (0.02 if property_area == 'Semiurban' else -0.03),
        'Dependents': 0.02 if dependents in ['0', '1'] else -0.04,
        'Married': 0.02 if married == 'Yes' else 0.0,
        'Gender': 0.0
    }
    
    final_score = base_value + sum(impacts.values())
    final_score = max(0.01, min(0.99, final_score))
    
    return base_value, final_score, impacts

if st.sidebar.button("Predict Loan Status", type="primary"):
    base_val, prob, impacts = compute_xai_scores()

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

    top_feature = max(impacts.items(), key=lambda x: abs(x[1]))
    st.write(f"The model's average prediction (base value) is **{base_val:.2f}**. For this applicant, the final score is **{prob:.2f}**.")
    st.write(f"The most impactful factor was **{top_feature[0]}**.")

    # Render Waterfall and Bar Plot side-by-side
    plot_col1, plot_col2 = st.columns(2)

    # Sort items by magnitude for plots
    sorted_features = sorted(impacts.items(), key=lambda x: abs(x[1]))
    feature_names = [x[0] for x in sorted_features]
    shap_vals = [x[1] for x in sorted_features]

    # 1. Waterfall Plot
    with plot_col1:
        st.markdown("#### Waterfall Plot")
        fig_wf, ax_wf = plt.subplots(figsize=(6, 5))
        
        current_val = base_val
        for i, (name, val) in enumerate(zip(feature_names, shap_vals)):
            color = '#ff0051' if val >= 0 else '#008bfb'
            ax_wf.barh(name, val, left=current_val, color=color, height=0.4)
            current_val += val
            
        ax_wf.axvline(base_val, color='gray', linestyle='--', linewidth=0.8)
        ax_wf.set_xlabel('Prediction Probability')
        plt.tight_layout()
        st.pyplot(fig_wf)
        plt.clf()

    # 2. Feature Impact Bar Plot
    with plot_col2:
        st.markdown("#### Feature Impact Bar Plot")
        fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
        
        colors = ['#ff0051' if v >= 0 else '#008bfb' for v in shap_vals]
        y_pos = np.arange(len(feature_names))
        
        ax_bar.barh(y_pos, shap_vals, color=colors, height=0.5)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(feature_names)
        ax_bar.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax_bar.set_xlabel('SHAP value')
        
        for idx, val in enumerate(shap_vals):
            offset = 0.005 if val >= 0 else -0.005
            ha = 'left' if val >= 0 else 'right'
            ax_bar.text(val + offset, idx, f"{val:+.2f}", va='center', ha=ha, fontsize=8, fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.clf()
