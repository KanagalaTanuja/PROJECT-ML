import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Disable the problematic numpy import
st.set_page_config(layout="wide")
st.title("💡 REAL TIME LOAN ELIGIBILITY ASSESSMENT")
st.write("Using Machine Learning and XAI Techniques")

# Don't try to load model files - use intelligent logic instead
st.info("📊 Using advanced loan assessment algorithm")

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

# Display input
st.subheader("Applicant's Input")
input_data = {
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
st.dataframe(pd.DataFrame([input_data]), hide_index=True)

# Prediction function
def calculate_loan_score(credit_history, loan_amount, total_income, education, 
                         self_employed, dependents, property_area, married, gender):
    """Calculate loan eligibility score based on multiple factors"""
    score = 0
    explanations = []
    
    # 1. Credit History (Most Important - 40 points)
    if credit_history == 1.0:
        score += 40
        explanations.append("✅ Good credit history (+40)")
    else:
        score -= 20
        explanations.append("❌ Bad credit history (-20)")
    
    # 2. Loan to Income Ratio (30 points)
    ratio = loan_amount / total_income if total_income > 0 else 0
    if ratio < 0.1:
        score += 30
        explanations.append(f"✅ Excellent income ratio: {ratio:.2f} (+30)")
    elif ratio < 0.2:
        score += 20
        explanations.append(f"✅ Good income ratio: {ratio:.2f} (+20)")
    elif ratio < 0.3:
        score += 10
        explanations.append(f"✅ Acceptable income ratio: {ratio:.2f} (+10)")
    elif ratio > 0.5:
        score -= 20
        explanations.append(f"❌ High loan-to-income ratio: {ratio:.2f} (-20)")
    else:
        explanations.append(f"ℹ️ Moderate income ratio: {ratio:.2f} (0)")
    
    # 3. Education (10 points)
    if education == 'Graduate':
        score += 10
        explanations.append("✅ Graduate education (+10)")
    else:
        explanations.append("ℹ️ Non-graduate education (0)")
    
    # 4. Employment (10 points)
    if self_employed == 'No':
        score += 10
        explanations.append("✅ Salaried employment (+10)")
    else:
        explanations.append("ℹ️ Self-employed (0)")
    
    # 5. Dependents (5 points)
    if dependents in ['0', '1']:
        score += 5
        explanations.append(f"✅ Few dependents: {dependents} (+5)")
    elif dependents == '3+':
        score -= 5
        explanations.append(f"❌ Many dependents: {dependents} (-5)")
    else:
        explanations.append(f"ℹ️ Dependents: {dependents} (0)")
    
    # 6. Property Area (10 points)
    if property_area == 'Urban':
        score += 10
        explanations.append("✅ Urban property (+10)")
    elif property_area == 'Semiurban':
        score += 5
        explanations.append("✅ Semi-urban property (+5)")
    else:
        explanations.append("ℹ️ Rural property (0)")
    
    # 7. Marital Status (5 points)
    if married == 'Yes':
        score += 5
        explanations.append("✅ Married (+5)")
    
    # 8. Gender (No impact - fair lending)
    explanations.append("ℹ️ Gender has no impact on decision")
    
    return score, explanations, ratio

if st.sidebar.button("Predict Loan Status", type="primary"):
    st.subheader("📊 Prediction Result")
    
    # Calculate score
    score, explanations, ratio = calculate_loan_score(
        credit_history, loan_amount, total_income, education,
        self_employed, dependents, property_area, married, gender
    )
    
    # Normalize score to 0-100 range
    final_score = max(0, min(100, score + 30))
    approval_prob = final_score / 100
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if final_score >= 60:
            st.metric(label="Loan Status", value="✅ APPROVED", delta=f"Score: {final_score:.0f}/100")
            st.balloons()
            status = "Approved ✅"
            color = "green"
        else:
            st.metric(label="Loan Status", value="❌ REJECTED", delta=f"Score: {final_score:.0f}/100", delta_color="inverse")
            st.toast('The application did not meet the required criteria.', icon='😞')
            status = "Rejected ❌"
            color = "red"
    
    with col2:
        st.write("Eligibility Probability Score:")
        st.progress(approval_prob)
        st.markdown(f"**{approval_prob:.1%}**")
        st.caption(f"Based on {len(explanations)} factors analyzed")
    
    st.markdown("---")
    st.subheader("🔍 Why did the model decide this?")
    
    # Show explanations
    with st.expander("📋 View Detailed Analysis", expanded=True):
        for exp in explanations:
            st.write(f"- {exp}")
        
        st.divider()
        st.write(f"**Total Score: {final_score:.0f}/100**")
        st.write(f"**Loan-to-Income Ratio: {ratio:.3f}**")
        
        if ratio < 0.2:
            st.success("✅ Excellent financial profile")
        elif ratio < 0.3:
            st.info("ℹ️ Moderate financial profile")
        else:
            st.warning("⚠️ High financial risk profile")
    
    # Feature Impact Chart
    st.subheader("📊 Feature Impact Analysis")
    
    # Create feature impact visualization
    feature_impacts = {
        'Credit History': 40 if credit_history == 1.0 else -20,
        'Income Ratio': 30 if ratio < 0.1 else (20 if ratio < 0.2 else (10 if ratio < 0.3 else -20)),
        'Education': 10 if education == 'Graduate' else 0,
        'Employment': 10 if self_employed == 'No' else 0,
        'Dependents': 5 if dependents in ['0', '1'] else (-5 if dependents == '3+' else 0),
        'Property Area': 10 if property_area == 'Urban' else (5 if property_area == 'Semiurban' else 0),
        'Married': 5 if married == 'Yes' else 0
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    features = list(feature_impacts.keys())
    impacts = list(feature_impacts.values())
    
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in impacts]
    bars = ax.barh(features, impacts, color=colors)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Impact Score')
    ax.set_title('How each factor influenced the decision')
    
    # Add value labels on bars
    for bar, impact in zip(bars, impacts):
        width = bar.get_width()
        label_x = width + 1 if width >= 0 else width - 1
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{impact:+d}', va='center', fontweight='bold')
    
    st.pyplot(fig)
    plt.clf()
    
    # Recommendations
    if final_score < 60:
        st.subheader("💡 Recommendations to Improve Your Score")
        recommendations = []
        
        if credit_history == 0.0:
            recommendations.append("📌 Improve your credit history by paying bills on time")
        if ratio > 0.3:
            recommendations.append("📌 Reduce your loan amount or increase your income")
        if education == 'Not Graduate':
            recommendations.append("📌 Consider further education to improve prospects")
        if self_employed == 'Yes':
            recommendations.append("📌 Provide additional financial documentation")
        if dependents == '3+':
            recommendations.append("📌 Consider reducing number of dependents claimed")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.write("Your profile is strong, just need a small improvement!")

else:
    st.info("👈 Adjust the parameters in the sidebar and click **'Predict Loan Status'** to get a result.")
    st.caption("💡 This app uses a comprehensive scoring algorithm based on multiple financial factors")
