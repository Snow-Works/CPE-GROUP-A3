
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline 
pipeline = joblib.load("laptop_pipeline.joblib")

st.title("Laptop Price Predictor")

# Input fields 
company = st.selectbox("Company", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "MSI", "Razer", "Other"])
typename = st.selectbox("Type", ["Notebook", "Gaming", "Ultrabook", "Workstation", "Convertible", "2 in 1 Convertible", "Other"])
inches = st.slider("Screen Size (inches)", 11.0, 20.0, 15.6)
ram = st.slider("RAM (GB)", 2, 64, 8)
opsys = st.selectbox("Operating System", ["Windows 10", "Windows 7", "Mac OS", "Linux", "No OS", "Other"])
weight = st.slider("Weight (kg)", 0.8, 5.0, 2.0)
touchscreen = st.radio("Touchscreen", ["Yes", "No"])
ips = st.radio("IPS Display", ["Yes", "No"])
x_res = st.number_input("X Resolution (px)", 800, 4000, 1366)
y_res = st.number_input("Y Resolution (px)", 600, 2500, 768)
cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD", "Other"])
cpu_speed = st.number_input("CPU Speed (GHz)", 1.0, 5.5, 2.5)
gpu_brand = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD", "Other"])

# Convert Yes/No → 0/1
touchscreen_val = 1 if touchscreen == "Yes" else 0
ips_val = 1 if ips == "Yes" else 0

# Auto-calculate PPI
ppi = ((x_res*2 + y_res*2) ** 0.5) / inches

# Build input dictionary 
user_input = {
    "Company": company,
    "TypeName": typename,
    "Inches": inches,
    "Ram": float(ram),
    "OpSys": opsys,
    "Weight": weight,
    "Touchscreen": touchscreen_val,
    "IPS": ips_val,
    "X_res": x_res,
    "Y_res": y_res,
    "PPI": ppi,
    "Cpu Brand": cpu_brand,
    "Cpu Speed": cpu_speed,
    "Gpu Brand": gpu_brand
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

st.write("### Your Laptop Specifications")
st.dataframe(input_df)

# Prediction button
if st.button("Predict Price"):
    try:
        price = pipeline.predict(input_df)[0]
        st.success(f"Estimated Laptop Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
