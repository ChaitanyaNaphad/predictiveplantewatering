import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os

# Title
st.title("🌱 Predictive Plant Watering App")





import pandas as pd

# dataset_url = ("https://raw.githubusercontent.com/ChaitanyaNaphad/chaitu_global/refs/heads/main/watering_schedule_combinations.csv"  )
dataset_url = "https://raw.githubusercontent.com/ChaitanyaNaphad/predictiveplantewatering/main/watering_schedule_combinations.csv"

df = pd.read_csv(dataset_url)


# File uploader for optional dataset upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

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
st.subheader("Select a plant for prediction:")
plant_options = {
    "Rubber plant": "Rubber Plant",
    "Coleus plant": "Coleus",
    "Polka dot plant": "Polka Dot Plant",
    "Dracaena plant": "Dracaena",
    "Polyscias plant": "Polyscias",
}
selected_plant = st.selectbox("Choose a plant", list(plant_options.keys()))

# User inputs
soil_moisture = st.number_input("Enter Soil Moisture (%)", min_value=0.0, max_value=100.0)
temperature = st.number_input("Enter Temperature (°C)", min_value=-10.0, max_value=50.0)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0)

if st.button("Predict Watering Time"):
    y = df[plant_options[selected_plant]]  # Get the correct target variable

    user_input = np.array([[soil_moisture, temperature, humidity]])
    user_input_poly = poly.transform(user_input)

    predicted_days = model.predict(user_input_poly)
    st.success(f"⏳ Recommended Watering in: {predicted_days[0]:.2f} days")
