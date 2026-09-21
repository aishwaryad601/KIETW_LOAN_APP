import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("model_lr.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details to predict the loan approval status.")


# Numerical inputs
dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=5,
    value=0,
    step=1
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0.0,
    value=5000.0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=0.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=100.0
)

loan_term = st.number_input(
    "Loan Term",
    min_value=0.0,
    value=360.0
)

credit_history = st.selectbox(
    "Credit History",
    [0, 1]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)


# Categorical inputs
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

employment_status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-Employed", "Unemployed"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)


# Convert categorical values into numerical values
gender_male = 1 if gender == "Male" else 0
married_yes = 1 if married == "Yes" else 0
education_not_graduate = 1 if education == "Not Graduate" else 0

employment_salaried = 1 if employment_status == "Salaried" else 0
employment_self_employed = 1 if employment_status == "Self-Employed" else 0
employment_unemployed = 1 if employment_status == "Unemployed" else 0

property_rural = 1 if property_area == "Rural" else 0
property_semiurban = 1 if property_area == "Semiurban" else 0
property_urban = 1 if property_area == "Urban" else 0


# Prediction
if st.button("🔮 Predict Loan Status"):

    input_data = np.array([[
        dependents,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        age,
        gender_male,
        married_yes,
        education_not_graduate,
        employment_salaried,
        employment_self_employed,
        employment_unemployed,
        property_rural,
        property_semiurban,
        property_urban
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Status: APPROVED")
    else:
        st.error("❌ Loan Status: REJECTED")