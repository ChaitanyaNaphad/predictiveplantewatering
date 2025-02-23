import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df =pd.read_csv("E:\\all_csv\\watering_schedule_combinations.csv")

def rubber_plant():

    X = df[["Soil Moisture (%)","Temperature (°C)","Humidity (%)"]]
    y = df["Rubber Plant"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_test)

    import pickle
    pickle.dump(user_input,open("vectorizer.pkl","wb"))
    pickle.dump(poly,open("model.pkl","wb"))

    # Apply the same polynomial transformation to user input
    user_input_poly = poly.transform(user_input)

    # Predict using trained model
    predicted_days = poly_reg.predict(user_input_poly)
    
    # Display prediction
    print(f"\n⏳ Recommended Watering in: {predicted_days[0]:.2f} days")


def coleus_plant():
    X = df[["Soil Moisture (%)","Temperature (°C)","Humidity (%)"]]
    y = df["Coleus"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_test)

    user_input_poly = poly.transform(user_input)
    import pickle
    pickle.dump(user_input,open("vectorizer.pkl","wb"))
    pickle.dump(poly,open("model.pkl","wb"))
    predicted_days = poly_reg.predict(user_input_poly)

    print(f"\n⏳ Recommended Watering in: {predicted_days[0]:.2f} days")


def polka_dot_plant():

    X = df[["Soil Moisture (%)","Temperature (°C)","Humidity (%)"]]
    y = df["Polka Dot Plant"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_test)

    user_input_poly = poly.transform(user_input)
    import pickle
    pickle.dump(user_input,open("vectorizer.pkl","wb"))
    pickle.dump(poly,open("model.pkl","wb"))
    predicted_days = poly_reg.predict(user_input_poly)

    print(f"\n⏳ Recommended Watering in: {predicted_days[0]:.2f} days")

def dracaena_plant():

    X = df[["Soil Moisture (%)","Temperature (°C)","Humidity (%)"]]
    y = df["Dracaena"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_test)

    user_input_poly = poly.transform(user_input)
    import pickle
    pickle.dump(user_input,open("vectorizer.pkl","wb"))
    pickle.dump(poly,open("model.pkl","wb"))
    predicted_days = poly_reg.predict(user_input_poly)

    print(f"\n⏳ Recommended Watering in: {predicted_days[0]:.2f} days")

def polyscias_plant():

    X = df[["Soil Moisture (%)","Temperature (°C)","Humidity (%)"]]
    y = df["Polyscias"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_test)

    user_input_poly = poly.transform(user_input)
    import pickle
    pickle.dump(user_input,open("vectorizer.pkl","wb"))
    pickle.dump(poly,open("model.pkl","wb"))
    predicted_days = poly_reg.predict(user_input_poly)

    print(f"\n⏳ Recommended Watering in: {predicted_days[0]:.2f} days")

print("1.Rubber plant")
print("2.Coleus plant")
print("3.Polka dot plant")
print("4.Dracaena plant")
print("5.Polyscias plant")
print("*******************")

soil_moisture = float(input("Enter Soil Moisture (%): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))

user_input = np.array([[soil_moisture, temperature, humidity]])
while True:
    user = int(input("Enter the plant number you want to check (or type '0' to exit): "))
    
    if user == 0:
        break
    
    if user==1:
        rubber_plant()
    elif user==2:
        coleus_plant()
    elif user==3:
        polka_dot_plant()
    elif user==4:
        dracaena_plant()
    elif user==5:
        polyscias_plant()
    else:
        print("Invalid input! Please enter a number between 1-5 or 'stop' to exit.")

