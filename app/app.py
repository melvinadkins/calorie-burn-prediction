# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.feature_engineering import FeatureEngineering

# Load XGBoost model
@st.cache_resource 
def load_model():
    #return joblib.load("../model/xgb_calories_model.joblib")
    model_path = Path(__file__).resolve().parents[1] / "model" / "xgb_calories_model.joblib"
    return joblib.load(model_path)

model = load_model()

# App Title 
st.title("🔥 Calorie Burn Prediction App")
st.write("Predict calories burned based on workout and physiological inputs.")

# Display image
#image = Image.open("../assets/gym.jpeg")
BASE_DIR = Path(__file__).resolve().parents[1]
image_path = BASE_DIR / "assets" / "gym.jpeg"
image = Image.open(image_path)
st.image(image)

st.divider()

# User inputs
sex = st.selectbox("Biological Sex", ["Male", "Female"])
age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
height_feet = st.number_input("Height (feet)", min_value=3, max_value=8, value=5)
height_inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=0)
weight_lbs = st.number_input("Weight (lbs)", min_value=50, max_value=400, value=170)
duration = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=30)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=200, value=120)
body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)

# Height in cm for prediction
height_cm = ((height_feet * 12) + height_inches) * 2.54

# Weight in kg for prediction
weight_kg = weight_lbs * 0.453592

# Convert sex to lowercase for prediction
sex_lowercase = sex.lower()

# Prediction
if st.button("Predict Calories Burned"):
    input_data = pd.DataFrame({
        "Sex": [sex_lowercase],
        "Age": [age],
        "Height": [height_cm],
        "Weight": [weight_kg],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Calories Burned: {round(prediction, 0)}")


# Contextual note
st.caption("ℹ️ This estimate is based on a machine learning model trained on exercise data. "
"Individual results may vary based on fitness level, exercise type, and other factors.")
