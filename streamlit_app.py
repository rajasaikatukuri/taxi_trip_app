import streamlit as st
import numpy as np
import pickle

st.image("taxi.jpg", width=400)

# Load models and scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

st.title("Taxi Tip Prediction App")

# Payment types with names (adjust as per your actual encoding)
payment_types = ["Select Payment Type", "Cash", "Credit Card", "Debit Card", "Prepaid Card", "Mobile Payment"]

# Input fields with min values and default to 0.0 (so + and - work)
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, value=0.0)
passenger_count = st.number_input("Passenger Count", min_value=1, step=1, value=1)
fare_amount = st.number_input("Fare Amount ($)", min_value=0.0, value=0.0)
extra = st.number_input("Extra ($)", min_value=0.0, value=0.0)
mta_tax = st.number_input("MTA Tax ($)", min_value=0.0, value=0.0)
tolls_amount = st.number_input("Tolls Amount ($)", min_value=0.0, value=0.0)
improvement_surcharge = st.number_input("Improvement Surcharge ($)", min_value=0.0, value=0.0)
total_amount = st.number_input("Total Amount ($)", min_value=0.0, value=0.0)

# Select payment type by name, default is 'Select Payment Type'
payment_type_name = st.selectbox("Payment Type", options=payment_types)

# Convert the payment type name to the appropriate encoding (you can adjust this based on how the model was trained)
payment_type_encoding = payment_types.index(payment_type_name) if payment_type_name != "Select Payment Type" else None

duration = st.number_input("Trip Duration (minutes)", min_value=0, value=0)

model_type = st.radio("Select Model", options=["Decision Tree", "Random Forest"])

if st.button("Predict Tip"):
    # Validation: Ensure all fields are filled
    if None in [trip_distance, passenger_count, fare_amount, extra, mta_tax, tolls_amount, 
                improvement_surcharge, total_amount, payment_type_encoding, duration] or payment_type_encoding is None:
        st.error("Please fill in all the fields before predicting.")
    else:
        features = np.array([[trip_distance, passenger_count, fare_amount, extra, mta_tax,
                              tolls_amount, improvement_surcharge, total_amount,
                              payment_type_encoding, duration]])
        scaled = scaler.transform(features)

        if model_type == "Decision Tree":
            prediction = dt_model.predict(scaled)[0]
        else:
            prediction = rf_model.predict(scaled)[0]

        st.success(f"Predicted Tip Amount: ${prediction:.2f}")




