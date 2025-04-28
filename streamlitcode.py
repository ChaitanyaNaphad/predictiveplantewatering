import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

# Title
st.title("🌱 Predictive Plant Watering App")

# Correct GitHub raw dataset URL
dataset_url = "https://raw.githubusercontent.com/ChaitanyaNaphad/predictiveplantewatering/main/watering_schedule_combinations.csv"

# Load dataset
try:
    df = pd.read_csv(dataset_url)
    available_plants = df.columns[3:]  # Extract plant names from dataset columns (skip first 3 sensor columns)
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    df = None

if df is not None:
    # Show available plants
    st.subheader("🪴 Available Plants for Prediction:")
    st.write(", ".join(available_plants))

    # Load or train model
    poly_path = "poly_transform.pkl"
    model_path = "model.pkl"

    if os.path.exists(poly_path) and os.path.exists(model_path):
        poly = pickle.load(open(poly_path, "rb"))
        model = pickle.load(open(model_path, "rb"))
    else:
        X = df[["Soil Moisture (%)", "Temperature (°C)", "Humidity (%)"]]
        y = df["Rubber Plant"]  # Defaulting to "Rubber Plant" model

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        with open(poly_path, "wb") as f:
            pickle.dump(poly, f)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # User Interface
    st.subheader("🌿 Select a plant for prediction:")
    selected_plant = st.selectbox("Choose a plant", available_plants)

    # User inputs
    soil_moisture = st.number_input("Enter Soil Moisture (%)", min_value=0.0, max_value=100.0)
    temperature = st.number_input("Enter Temperature (°C)", min_value=-10.0, max_value=50.0)
    humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0)

    # Prediction function
    def predict_watering(plant_name):
        X = df[["Soil Moisture (%)", "Temperature (°C)", "Humidity (%)"]]
        y = df[plant_name]

        X_poly = poly.fit_transform(X)
        model.fit(X_poly, y)

        user_input = np.array([[soil_moisture, temperature, humidity]])
        user_input_poly = poly.transform(user_input)

        predicted_days = model.predict(user_input_poly)
        return predicted_days[0]

    # Prediction button
    if st.button("Predict Watering Days"):
        output = ""

        # Check soil moisture condition and display appropriate warning
        if soil_moisture < 30:
            output += "⚠️ Soil moisture is below 30%. Please water your plant immediately!\n\n"
        elif soil_moisture > 60:
            output += "⚠️ Soil is over moist. Watering is not necessary at the moment.\n\n"

        # Only calculate and display watering days if moisture >= 30
        if soil_moisture >= 30:
            watering_days = predict_watering(selected_plant)

            # Convert decimal days into days and hours
            days = int(watering_days)
            hours = int((watering_days - days) * 24)

            output += f"⏳ Recommended Watering in: {days} days and {hours} hours"

        # Display the output below the button
        st.text_area("", output, height=200)

    # Footer
    st.markdown("---")
    st.markdown("Created by [Chaitanya Naphad] 🌱")
