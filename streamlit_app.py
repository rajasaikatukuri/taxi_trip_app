import streamlit as st
import numpy as np
import pickle
import random

# Load models and scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Title and image
st.image("taxi.jpg", width=400)
st.title("🚕 NYC Taxi Tip Prediction App")

# Defaults (can be replaced when "Generate Random Trip" is clicked)
default_inputs = {
    "trip_distance": 2.5,
    "passenger_count": 1,
    "fare_amount": 15.0,
    "extra": 1.0,
    "mta_tax": 0.5,
    "tolls_amount": 5.0,
    "improvement_surcharge": 0.3,
    "duration": 15,
    "payment_type_encoding": 1,
}

# 🎲 Random Trip Generation
if st.button("🎲 Generate Random Trip"):
    default_inputs["trip_distance"] = round(np.random.uniform(0.5, 10), 2)
    default_inputs["passenger_count"] = np.random.randint(1, 5)
    default_inputs["fare_amount"] = round(np.random.uniform(5, 50), 2)
    default_inputs["extra"] = round(random.choice([0.5, 1.0, 2.5]), 2)
    default_inputs["mta_tax"] = 0.5
    default_inputs["tolls_amount"] = round(np.random.uniform(0, 15), 2)
    default_inputs["improvement_surcharge"] = 0.3
    default_inputs["duration"] = np.random.randint(5, 45)
    default_inputs["payment_type_encoding"] = np.random.randint(1, 6)

# Input Section
with st.expander("🚗 Trip & Fare Details", expanded=True):
    trip_distance = st.slider("Trip Distance (miles)", 0.0, 20.0, step=0.1, value=default_inputs["trip_distance"])
    passenger_count = st.slider("Passenger Count", 1, 6, value=default_inputs["passenger_count"])
    duration = st.slider("Trip Duration (minutes)", 1, 120, value=default_inputs["duration"])

    fare_amount = st.slider("Fare Amount ($)", 0.0, 100.0, step=1.0, value=default_inputs["fare_amount"])
    extra = st.slider("Extra Charges ($)", 0.0, 10.0, step=0.5, value=default_inputs["extra"])
    mta_tax = st.slider("MTA Tax ($)", 0.0, 1.0, step=0.5, value=default_inputs["mta_tax"])
    tolls_amount = st.slider("Tolls Amount ($)", 0.0, 20.0, step=0.5, value=default_inputs["tolls_amount"])
    improvement_surcharge = st.slider("Improvement Surcharge ($)", 0.0, 2.0, step=0.1, value=default_inputs["improvement_surcharge"])

# Auto calculate total
total_amount = fare_amount + extra + mta_tax + tolls_amount + improvement_surcharge
st.markdown(f"### 💵 Total Amount: `{total_amount:.2f} USD`")

# Payment options
payment_display = ["💵 Cash", "💳 Credit Card", "🏦 Debit Card", "🧾 Prepaid Card", "📱 Mobile Payment"]
payment_encoding = {
    "💵 Cash": 1,
    "💳 Credit Card": 2,
    "🏦 Debit Card": 3,
    "🧾 Prepaid Card": 4,
    "📱 Mobile Payment": 5
}
selected_payment = st.radio("💳 Payment Method", payment_display, index=default_inputs["payment_type_encoding"] - 1)
payment_type_encoding = payment_encoding[selected_payment]

# Model choice
model_type = st.radio("🧠 Select Model", options=["Decision Tree", "Random Forest"])

# Prediction
if st.button("🚀 Predict Tip"):
    features = np.array([[trip_distance, passenger_count, fare_amount, extra, mta_tax,
                          tolls_amount, improvement_surcharge, total_amount,
                          payment_type_encoding, duration]])
    scaled = scaler.transform(features)

    if model_type == "Decision Tree":
        prediction = dt_model.predict(scaled)[0]
    else:
        prediction = rf_model.predict(scaled)[0]

    st.success(f"🎯 Estimated Tip: **${prediction:.2f}**")
