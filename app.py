import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Flight Fare Predictor", layout="centered")

# =========================
# CUSTOM CSS (FULL UI)
# =========================
st.markdown("""
<style>

/* Page background */
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

/* Title */
.title {
    font-size: 28px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    font-size: 14px;
    opacity: 0.9;
}

/* Author badge */
.author {
    margin-top: 15px;
    display: inline-block;
    background: rgba(255,255,255,0.25);
    padding: 10px 18px;
    border-radius: 25px;
    font-size: 14px;
    font-weight: 500;
    backdrop-filter: blur(5px);
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 0 0 20px 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.1);
}

/* Section title */
.section-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #3b82f6, #7c3aed);
    color: white;
    font-weight: bold;
    height: 45px;
    border: none;
}

/* Input spacing */
.stSelectbox, .stNumberInput, .stTextInput, .stDateInput, .stTimeInput {
    margin-bottom: 12px;
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
    <div class="author">👤 Created by Amol Satpute</div>
</div>
""", unsafe_allow_html=True)

# =========================
# CARD START
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">🛫 Flight Information</div>', unsafe_allow_html=True)

# =========================
# INPUT FIELDS
# =========================
airline = st.selectbox("✈ Airline", ["IndiGo", "Air India", "SpiceJet", "Vistara"])
travel_class = st.selectbox("💺 Travel Class", ["Economy", "Business"])
stops = st.selectbox("🔁 Stops", [0, 1, 2, 3])
duration = st.number_input("⏱ Duration (minutes)", 30, 2000)

flight_no = st.text_input("🆔 Flight Number", "FL-001")

date = st.date_input("📅 Journey Date")
dep_time = st.time_input("🕐 Departure Time")
arrival_time = st.time_input("🕒 Arrival Time")

source = st.selectbox("📍 Source", ["Delhi", "Mumbai", "Kolkata", "Chennai"])
destination = st.selectbox("📍 Destination", ["Cochin", "Hyderabad", "Banglore", "Delhi"])

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict Flight Fare"):

    data = np.zeros(len(columns))
    df_input = pd.DataFrame([data], columns=columns)

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

# =========================
# FOOTER
# =========================
st.markdown("""
<br>
<div style='text-align:center; font-size:13px; color:gray;'>
© 2026 Amol Satpute | Flight Fare Prediction System
</div>
""", unsafe_allow_html=True)
