import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium  # Replacing folium_static
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load trained model
with open("FinalModel.xgb", "rb") as f:
    model = xgb.XGBRegressor()
    model.load_model("FinalModel.xgb")  # ✅ Correct way to load XGBoost model


# Load dataset used for training (to extract expected feature names)
with open("train_columns (1).pkl", "rb") as f:
    train_columns = pickle.load(f)  # List of features used during training

# Streamlit App Title
st.title("House Price Prediction App")

# Sidebar Inputs
st.sidebar.header("User Input Features")
latitude = st.sidebar.number_input("Latitude", value=37.77, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-122.42, format="%.6f")
overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
garage_cars = st.sidebar.slider("Garage Cars", 0, 5, 2)
garage_area = st.sidebar.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=400)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
year_remod_add = st.sidebar.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2005)

# Neighborhood selection
neighborhoods = ["Downtown", "Suburban", "Rural"]
neighborhood = st.sidebar.selectbox("Neighborhood Type", neighborhoods)

# Additional Inputs
crime_rate = st.sidebar.number_input("Crime Rate (per 1000 people)", min_value=0.0, max_value=100.0, value=5.0)
distance_to_school = st.sidebar.number_input("Distance to School (miles)", min_value=0.1, max_value=10.0, value=2.0)
population_density = st.sidebar.number_input("Population Density (people/sq mile)", min_value=100, max_value=5000, value=1200)
median_income = st.sidebar.number_input("Median Household Income ($)", min_value=10000, max_value=200000, value=60000)

# Create DataFrame for input
input_data = {
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "TotalBsmtSF": total_bsmt_sf,
    "GarageCars": garage_cars,
    "GarageArea": garage_area,
    "YearBuilt": year_built,
    "YearRemodAdd": year_remod_add,
    "Latitude": latitude,
    "Longitude": longitude,
    "neighborhood_Downtown": 1 if neighborhood == "Downtown" else 0,
    "crime_rate": crime_rate,
    "distance_to_school": distance_to_school,
    "population_density": population_density,
    "median_income": median_income,
}

input_df = pd.DataFrame([input_data])

# Ensure feature consistency
input_df = input_df.reindex(columns=train_columns, fill_value=0)  # Align with training features

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

# Map Display
st.subheader("Selected Location on Map")
m = folium.Map(location=[latitude, longitude], zoom_start=12)
folium.Marker([latitude, longitude], popup="Selected Location").add_to(m)

# Display map using new `st_folium`
st_folium(m, width=700, height=500)
