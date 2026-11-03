
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("soh_model.pkl")

# Page configuration
st.set_page_config(
    page_title="EV Battery Health Prediction",
    page_icon="🔋",
    layout="centered"
)

# Application heading
st.title("🔋 EV Battery Health Prediction")

st.write(
    "Predict Battery State of Health (SOH) "
    "from the number of charging cycles."
)

st.divider()

# User input
cycles = st.number_input(
    "Enter Charging Cycles:",
    min_value=0,
    max_value=2000,
    value=850,
    step=50
)

# Prediction button
if st.button("Predict Battery Health"):
    prediction = model.predict(
        np.array([[cycles]])
    )[0]

    # Keep prediction within valid SOH range
    prediction = np.clip(prediction, 0, 100)

    st.subheader("Predicted Battery Health:")

    st.success(f"{prediction:.2f} %")

    if prediction >= 80:
        status = "Good"
    elif prediction >= 70:
        status = "Moderate"
    else:
        status = "Poor"

    st.write(f"**Battery Health Status:** {status}")

    st.progress(int(prediction))

    st.caption(
        "Prediction generated using Single Linear Regression."
    )
