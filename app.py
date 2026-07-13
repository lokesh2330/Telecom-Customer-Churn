import streamlit as st
import pandas as pd
import joblib


# Load Model
model = joblib.load(
    "Models/churn_model.pkl"
)


# Page Title
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>🏢Telecom Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "Enter customer details and click **Predict** to check churn probability."
)


# Customer Details
col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=1,
        max_value=72,
        value=1
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=18.00,
        max_value=350.00,
        value=50.00
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=(monthly_charges*tenure),
        value=monthly_charges*tenure
    )
    
with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    if internet_service == "No":
        online_security = st.selectbox(
            "Online Security",
            ["No internet service"],
            disabled=True
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No internet service"],
            disabled=True
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No internet service"],
            disabled=True
        )

    else:
            online_security = st.selectbox(
            "Online Security",
            ["No", "Yes"]
        )

            online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes"]
        )

            tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes"]
        )

     

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

    

# Encoding Dictionaries
gender_map = {
    "Female": 0,
    "Male": 1
}

yes_no_map = {
    "No": 0,
    "Yes": 1
}


internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

service_map = {
    "No": 0,
    "No internet service": 1,
    "Yes": 2
}

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3
}

#st.write(list(model.feature_names_in_))

# Predict Button
if st.button("🔍 Predict Churn"):


    input_data = pd.DataFrame([{
        "gender": gender_map[gender],
        "SeniorCitizen": yes_no_map[senior],
        "Partner": yes_no_map[partner],
        "Dependents": yes_no_map[dependents],
        "tenure": tenure,
        "InternetService": internet_map[internet_service],
        "Contract": contract_map[contract],
        "PaymentMethod": payment_map[payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "OnlineSecurity": service_map[online_security],
        "TechSupport": service_map[tech_support],
        "OnlineBackup": service_map[online_backup]
        
    }])
    if tenure <= 0:
        st.error("❌ Tenure must be greater than 0 months.")
        st.stop()

    if monthly_charges <= 0:
        st.error("❌ Monthly Charges must be greater than 0.")
        st.stop()

    if total_charges < (monthly_charges*tenure):
        st.error(
            f"❌ Total Charges should be at least ₹500.\n\n"
            f"Current Total Charges: ₹{total_charges:.2f}"
        )
        st.stop()

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    stay_probability = probabilities[0]
    churn_probability = probabilities[1]

    st.markdown("---")

    st.subheader("Customer Details")
    st.write(f"Gender : **{gender}**")
    st.write(f"Senior Citizen : **{senior}**")
    st.write(f"Partner : **{partner}**")
    st.write(f"Dependents : **{dependents}**")
    st.write(f"Tenure : **{tenure}** months")
    st.write(f"Internet Service : **{internet_service}**")
    st.write(f"Contract : **{contract}**")
    st.write(f"Payment Method : **{payment_method}**")
    st.write(f"Monthly Charges : ₹**{monthly_charges:.2f}**")
    st.write(f"Total Charges : ₹**{total_charges:.2f}**")

    if prediction == 1:
        st.error("⚠ Customer is likely to CHURN")
        if churn_probability >= 0.80:
            st.warning("High Risk Customer")
        else:
            st.info("Moderate Risk Customer")
    else:
        st.success("Low Risk Customer")
        st.success("✅ Customer is likely to STAY")

    st.write(f"**Churn Probability:** {churn_probability:.2%}")
    st.progress(float(churn_probability))
    st.write(f"**Stay Probability:** {stay_probability:.2%}")