import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Import the Firebase data retrieval function
from firebase_config import get_sensor_data

# Load dataset from GitHub
dataset_url = "https://raw.githubusercontent.com/ChaitanyaNaphad/predictiveplantewatering/refs/heads/main/watering_schedule_combinations.csv"
df = pd.read_csv(dataset_url)

# Streamlit UI
st.set_page_config(page_title="Plant Watering Predictor", layout="centered")

st.title("🌿 Plant Watering Predictor")
st.subheader("Predict when your plant needs watering based on environmental conditions.")

# Sidebar: Choose input mode
input_mode = st.radio("Select Input Mode:", ("Manual Input", "Use Firebase Sensor Data"))

if input_mode == "Manual Input":
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    user_input = np.array([[soil_moisture, temperature, humidity]])
else:
    st.write("Fetching sensor data from Firebase...")
    sensor_data = get_sensor_data()
    if sensor_data:
        temperature = sensor_data.get("Temperature", 25.0)
        humidity = sensor_data.get("Humidity", 50.0)
        soil_moisture = sensor_data.get("SoilMoisture", 50.0)
        st.write(f"🌡️ Temperature: {temperature} °C")
        st.write(f"💧 Humidity: {humidity} %")
        st.write(f"🌱 Soil Moisture: {soil_moisture} %")
        user_input = np.array([[soil_moisture, temperature, humidity]])
    else:
        st.error("No sensor data available. Using default values.")
        user_input = np.array([[50.0, 25.0, 50.0]])

# Display available plants
st.subheader("🌱 Available Plants")
plant_options = {
    "Rubber Plant": "Rubber Plant",
    "Coleus": "Coleus",
    "Polka Dot Plant": "Polka Dot Plant",
    "Dracaena": "Dracaena",
    "Polyscias": "Polyscias"
}
st.write(", ".join(plant_options.keys()))

# Plant selection dropdown
selected_plant = st.selectbox("Select a plant:", list(plant_options.keys()))

# Prediction function
def predict_watering(plant_name):
    X = df[["Soil Moisture (%)", "Temperature (°C)", "Humidity (%)"]]
    y = df[plant_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)

    user_input_poly = poly.transform(user_input)
    predicted_days = poly_reg.predict(user_input_poly)

    return predicted_days[0]

# Prediction button
if st.button("Predict Watering Days"):
    watering_days = predict_watering(plant_options[selected_plant])
    st.success(f"⏳ Recommended Watering in: {watering_days:.2f} days")

# Footer
st.markdown("---")
st.markdown("Created by [Chaitanya Naphad] 🌱")
