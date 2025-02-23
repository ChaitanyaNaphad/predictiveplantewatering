import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load dataset
# dataset_url = ("https://raw.githubusercontent.com/ChaitanyaNaphad/chaitu_global/refs/heads/main/watering_schedule_combinations.csv")
# df = pd.read_csv(dataset_url)
df =pd.read_csv("E:\\all_csv\\watering_schedule_combinations.csv")
# Streamlit UI with dark theme
st.set_page_config(page_title="Plant Watering Predictor", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stSlider {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Plant Watering Predictor")
st.write("Enter environmental conditions to get watering recommendations.")

# List available plants
st.subheader("🌱 Available Plants")
plant_options = {
    "Rubber Plant": "Rubber Plant",
    "Coleus": "Coleus",
    "Polka Dot Plant": "Polka Dot Plant",
    "Dracaena": "Dracaena",
    "Polyscias": "Polyscias"
}
for plant in plant_options.keys():
    st.write(f"- {plant}")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")
soil_moisture = st.sidebar.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
user_input = np.array([[soil_moisture, temperature, humidity]])

# Plant selection dropdown
selected_plant = st.selectbox("Select a plant:", list(plant_options.keys()))

def predict_watering(plant_name):
    X = df[["Soil Moisture (%)", "Temperature (°C)", "Humidity (%)"]]
    y = df[plant_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    
    user_input_poly = poly.transform(user_input)
    predicted_days = poly_reg.predict(user_input_poly)
    
    return predicted_days[0]

if st.button("Predict Watering Days"):
    watering_days = predict_watering(plant_options[selected_plant])
    st.success(f"⏳ Recommended Watering in: {watering_days:.2f} days")
    
# Footer
st.markdown("---")
st.markdown("Created by [Chaitanya Naphad] 🌱")
