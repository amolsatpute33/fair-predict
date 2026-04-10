import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Flight Fare Predictor", layout="centered")

# =========================
# CUSTOM CSS (MAIN DESIGN)
# =========================
st.markdown("""
<style>
/* Background */
body {
    background-color: #f5f7fb;
}

/* Header */
.header {
    background: linear-gradient(135deg, #3b82f6, #7c3aed);
    padding: 30px;
    border-radius: 20px 20px 0 0;
    color: white;
    text-align: center;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 0 0 20px 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.1);
}

/* Title */
.title {
    font-size: 26px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    font-size: 14px;
    opacity: 0.9;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #3b82f6, #7c3aed);
    color: white;
    font-weight: bold;
    height: 45px;
}

/* Input spacing */
.stSelectbox, .stNumberInput, .stTextInput {
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header">
    <div class="title">✈️ Flight Fare Predictor</div>
    <div class="subtitle">Get accurate flight price predictions powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# =========================
# CARD START
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🛫 Flight Information")

# Inputs (same as UI)
airline = st.selectbox("✈ Airline", ["IndiGo", "Air India", "SpiceJet", "Vistara"])
travel_class = st.selectbox("💺 Travel Class", ["Economy", "Business"])
stops = st.selectbox("🔁 Stops", [0, 1, 2, 3])
duration = st.number_input("⏱ Duration (minutes)", 30, 2000)

flight_no = st.text_input("🆔 Flight Number", "FL-001")

# Dummy time inputs (for model compatibility)
date = st.date_input("📅 Journey Date")
dep_time = st.time_input("🕐 Departure Time")
arrival_time = st.time_input("🕒 Arrival Time")

source = st.selectbox("📍 Source", ["Delhi", "Mumbai", "Kolkata", "Chennai"])
destination = st.selectbox("📍 Destination", ["Cochin", "Hyderabad", "Banglore", "Delhi"])

# =========================
# PREDICT BUTTON
# =========================
if st.button("🔮 Predict Flight Fare"):

    data = np.zeros(len(columns))
    df_input = pd.DataFrame([data], columns=columns)

    # Fill values
    df_input["Journey_day"] = date.day
    df_input["Journey_month"] = date.month
    df_input["Dep_hour"] = dep_time.hour
    df_input["Dep_min"] = dep_time.minute
    df_input["Arrival_hour"] = arrival_time.hour
    df_input["Arrival_min"] = arrival_time.minute
    df_input["Duration"] = duration
    df_input["Total_Stops"] = stops

    # One-hot encoding
    if f"Airline_{airline}" in df_input.columns:
        df_input[f"Airline_{airline}"] = 1

    if f"Source_{source}" in df_input.columns:
        df_input[f"Source_{source}"] = 1

    if f"Destination_{destination}" in df_input.columns:
        df_input[f"Destination_{destination}"] = 1

    prediction = model.predict(df_input)[0]

    st.success(f"💰 Estimated Price: ₹ {int(prediction)}")

st.markdown('</div>', unsafe_allow_html=True)

