import streamlit as st
import joblib
import numpy as np

# Load the model saved by Member 3
# Note: If Member 3 named the file differently, update the filename here
try:
    model = joblib.load('student_budget_model.pkl')
except:
    st.error("Error: 'student_budget_model.pkl' not found. Ensure Member 3 has run the training script.")

# Page Configuration
st.set_page_config(page_title="Budget Predictor", page_icon="💰", layout="wide")

# Custom CSS to make it look less 'AI-ish' and more like a student project
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_index=True)

st.title("🎓 Student Budget Management Predictor")
st.write("This tool uses Machine Learning to analyze your spending habits and predict if you can manage your budget effectively until the end of the month.")

st.sidebar.header("Instructions")
st.sidebar.info("Enter your monthly financial details in the main panel. The model will calculate the risk based on patterns learned from 170+ student responses.")

# Create input fields for the 9 features used in training
col1, col2 = st.columns(2)

with col1:
    st.subheader("Income & Essentials")
    income = st.number_input("Monthly Income / Allowance (₹)", min_value=0, value=5000, step=500)
    rent = st.number_input("Rent / Hostel Fee (₹)", min_value=0, value=0, step=500)
    food = st.number_input("Monthly Food Spending (₹)", min_value=0, value=2000, step=200)
    transport = st.number_input("Transportation Costs (₹)", min_value=0, value=500, step=100)
    mobile = st.number_input("Mobile & Internet Recharge (₹)", min_value=0, value=400, step=50)

with col2:
    st.subheader("Lifestyle & Savings")
    edu = st.number_input("Education (Books/Prints) (₹)", min_value=0, value=500, step=100)
    eating_out = st.number_input("Eating Out / Delivery (₹)", min_value=0, value=1000, step=200)
    shopping = st.number_input("Shopping & Entertainment (₹)", min_value=0, value=1000, step=200)
    savings = st.number_input("Average Monthly Savings (₹)", min_value=0, value=500, step=100)

st.write("---")

# Prediction Logic
if st.button("Generate Financial Risk Report"):
    # The order MUST match the 'features' list in Member 3's code
    features = np.array([[income, rent, food, transport, mobile, edu, eating_out, shopping, savings]])
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0] # [Prob of 0, Prob of 1]
    
    risk_score = probability[1] * 100
    
    st.subheader("Analysis Results:")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk:** Based on your spending, you are likely to run out of money before the month ends.")
        st.metric(label="Financial Risk Level", value=f"{risk_score:.1f}%", delta="High", delta_color="inverse")
    else:
        st.success(f"✅ **Safe:** Your budget management seems stable for the month.")
        st.metric(label="Financial Risk Level", value=f"{risk_score:.1f}%", delta="Low", delta_color="normal")

    # Add a simple chart to show risk
    st.progress(int(risk_score))
    
    # Simple advice based on inputs
    if eating_out + shopping > income * 0.4:
        st.warning("💡 **Suggestion:** Your discretionary spending (Eating Out + Shopping) is quite high compared to your income. Reducing this could lower your risk.")
