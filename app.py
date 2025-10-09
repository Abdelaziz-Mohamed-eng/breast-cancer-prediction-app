import streamlit as st
import numpy as np
import joblib

# ==============================
# Load model and scaler
# ==============================
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🧬")
st.title("🩺 Breast Cancer Prediction App")
st.write("Enter the medical measurements below to predict whether the tumor is **Benign (0)** or **Malignant (1)**.")

# ==============================
# Input fields (10 important features)
# ==============================
st.subheader("🔹 Input Features")

radius_mean = st.number_input("Mean Radius", min_value=0.0)
texture_mean = st.number_input("Mean Texture", min_value=0.0)
perimeter_mean = st.number_input("Mean Perimeter", min_value=0.0)
area_mean = st.number_input("Mean Area", min_value=0.0)
concavity_mean = st.number_input("Mean Concavity", min_value=0.0)
concave_points_mean = st.number_input("Mean Concave Points", min_value=0.0)
radius_worst = st.number_input("Worst Radius", min_value=0.0)
perimeter_worst = st.number_input("Worst Perimeter", min_value=0.0)
area_worst = st.number_input("Worst Area", min_value=0.0)
concave_points_worst = st.number_input("Worst Concave Points", min_value=0.0)

# ==============================
# Prediction button
# ==============================
if st.button("🔍 Predict"):
    # Prepare input as a single row
    features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                          concavity_mean, concave_points_mean,
                          radius_worst, perimeter_worst, area_worst, concave_points_worst]])
    
    # Apply scaling
    features_scaled = scaler.transform(features)
    
    # Predict using model
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    # ==============================
    # Output result
    # ==============================
    st.subheader("📊 Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ The tumor is likely **Malignant (Cancerous)**\n\n🔸 Probability: {probability:.2f}")
    else:
        st.success(f"✅ The tumor is likely **Benign (Non-cancerous)**\n\n🔹 Probability: {probability:.2f}")
