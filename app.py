import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Streamlit app title
st.title("🏠 House Price Prediction")

# Sidebar for user inputs
st.sidebar.header("Input House Details")

# Function to extract features from location (mock implementation)
def extract_features_from_location(latitude, longitude):
    """
    Mock function to simulate feature extraction based on location.
    Replace this with actual API calls or dataset lookups.
    """
    features = {
        "neighborhood": "Downtown",  # Example neighborhood
        "crime_rate": 0.05,  # Example crime rate
        "distance_to_school": 2.5,  # Example distance to school in km
        "population_density": 5000,  # Example population density
        "median_income": 75000,  # Example median income
        "num_bedrooms": 3,  # Example number of bedrooms
        "num_bathrooms": 2,  # Example number of bathrooms
        "square_footage": 1800,  # Example square footage
    }
    return features

# Function to get user inputs
def get_user_inputs():
    # Numerical inputs
    overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sqft)", min_value=0, value=1500)
    total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sqft)", min_value=0, value=1000)
    garage_cars = st.sidebar.slider("Garage Cars", 0, 4, 2)
    garage_area = st.sidebar.number_input("Garage Area (sqft)", min_value=0, value=500)
    year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2023, value=2000)
    year_remod_add = st.sidebar.number_input("Year Remodeled", min_value=1800, max_value=2023, value=2000)

    # Categorical inputs
    ms_zoning = st.sidebar.selectbox("MS Zoning", ["RL", "RM", "C (all)", "FV", "RH"])
    lot_config = st.sidebar.selectbox("Lot Configuration", ["Inside", "Corner", "CulDSac", "FR2", "FR3"])
    house_style = st.sidebar.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer"])

    # Map for location selection
    st.sidebar.header("Select Location on Map")
    st.sidebar.write("Click on the map to select a location.")

    # Initialize latitude and longitude in session state
    if "latitude" not in st.session_state:
        st.session_state.latitude = 40.7128  # Default latitude (New York)
    if "longitude" not in st.session_state:
        st.session_state.longitude = -74.0060  # Default longitude (New York)

    # Create a folium map
    m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=11)

    # Add a click event to the map
    m.add_child(folium.LatLngPopup())

    # Display the map
    map_data = folium_static(m)

    # Update latitude and longitude based on map click
    if map_data:
        if "last_clicked" in st.session_state:
            st.session_state.latitude = st.session_state.last_clicked["lat"]
            st.session_state.longitude = st.session_state.last_clicked["lng"]

    # Display selected latitude and longitude
    st.sidebar.write(f"Selected Latitude: {st.session_state.latitude}")
    st.sidebar.write(f"Selected Longitude: {st.session_state.longitude}")

    # Extract features from location
    location_features = extract_features_from_location(st.session_state.latitude, st.session_state.longitude)

    # Create a dictionary of user inputs
    user_inputs = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod_add,
        "MSZoning": ms_zoning,
        "LotConfig": lot_config,
        "HouseStyle": house_style,
        "Latitude": st.session_state.latitude,
        "Longitude": st.session_state.longitude,
        **location_features,  # Add extracted location features
    }

    return user_inputs

# Get user inputs
user_inputs = get_user_inputs()

# Convert user inputs into a DataFrame
input_df = pd.DataFrame([user_inputs])

# One-hot encode categorical variables
input_df = pd.get_dummies(input_df)

# Ensure the input DataFrame has the same columns as the training data
# (You need to align the columns with the training data)
# This is a placeholder; you need to adjust it based on your actual training data
training_columns = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "GarageArea", 
    "YearBuilt", "YearRemodAdd", "MSZoning_RL", "MSZoning_RM", "MSZoning_C (all)", 
    "MSZoning_FV", "MSZoning_RH", "LotConfig_Inside", "LotConfig_Corner", 
    "LotConfig_CulDSac", "LotConfig_FR2", "LotConfig_FR3", "HouseStyle_1Story", 
    "HouseStyle_2Story", "HouseStyle_1.5Fin", "HouseStyle_1.5Unf", "HouseStyle_SFoyer", 
    "Latitude", "Longitude", "neighborhood_Downtown", "crime_rate", "distance_to_school", 
    "population_density", "median_income", "num_bedrooms", "num_bathrooms", "square_footage"
]

# Add missing columns to the input DataFrame
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match the training data
input_df = input_df[training_columns]

# Predict the house price
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

# Display user inputs
st.subheader("User Inputs")
st.write(input_df)