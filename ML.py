# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:21:53 2024

@author: Jai
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
from tqdm import tqdm

# Set page configuration
st.set_page_config(page_title="Climate Change Impact ML", page_icon="🌍")
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6Zpewx4TRzCvCiPfVJixgBPZyKERGlvXrHQ&s") no-repeat center center fixed;
        
        background-size: cover;
    
        
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load your data
data = pd.read_csv('C://Users//Jai//Downloads//POWER_Regional_Monthly_1984_2022.csv')

# Placeholder for machine learning model integration
st.subheader("Machine Learning Predictions On Climate Change")
st.write("Select a location on the map to view predictions.")

# Ensure LAT and LON are numeric
data['LAT'] = pd.to_numeric(data['LAT'], errors='coerce')
data['LON'] = pd.to_numeric(data['LON'], errors='coerce')

# Load cached location names
location_cache = pd.read_csv('location_cache.csv')

# Map location names back to the dataframe
data = data.merge(location_cache, on=['LAT', 'LON'], how='left')

# Sidebar widgets for user input
parameter = st.sidebar.selectbox("Select Parameter", data['PARAMETER'].unique())

# Filter data based on selected parameter
filtered_data = data[data['PARAMETER'] == parameter]

# Ensure that the filtered data contains entries for all months
months = [col for col in filtered_data.columns if col not in ['PARAMETER', 'YEAR', 'LAT', 'LON', 'ANN']]

# Melt data for better handling, excluding 'ANN'
melted_data = pd.melt(filtered_data, id_vars=['LAT', 'LON', 'Location'], 
                      value_vars=months,
                      var_name='Month', value_name='Value')

# Create map plot
fig = px.scatter_mapbox(melted_data, lat='LAT', lon='LON', hover_name='Location',
                        hover_data={'Value': True, 'Month': True},
                        zoom=3, height=600, color='Month')

fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)

# Data Preprocessing for ML Model
melted_data['Month'] = pd.Categorical(melted_data['Month'], categories=months, ordered=True)
melted_data['Month'] = melted_data['Month'].cat.codes + 1  # Convert month names to numerical values

# Features and target variable
X = melted_data[['LAT', 'LON', 'Month']]
y = melted_data['Value']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Mean Squared Error: {mse}")



# Event handling for map click or hover using st.query_params
query_params = st.experimental_get_query_params()
if 'lat' in query_params and 'lon' in query_params:
    lat = float(query_params['lat'][0])
    lon = float(query_params['lon'][0])

    # Placeholder for machine learning model prediction
    month = 1  # Default month (January)
    if 'month' in query_params:
        month = int(query_params['month'][0])

    # Make prediction
    predicted_value = model.predict(np.array([[lat, lon, month]]))[0]
    
    # Display the prediction information
    prediction_text = f"Predicted {parameter} at {lat}, {lon} for month {month}: {predicted_value:.2f}"
    st.write(prediction_text)

# Function to get location name from coordinates with retry mechanism
geolocator = Nominatim(user_agent="geoapiExercises", timeout=10)

def get_location_name(lat, lon, retries=3):
    for attempt in range(retries):
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True)
            return location.address if location else None
        except GeocoderTimedOut:
            if attempt < retries - 1:
                print(f"Timeout, retrying ({attempt+1}/{retries})...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                return "Timeout: Unable to fetch location"
        except Exception as e:
            print(f"Error: {e}")
            return "Error: Unable to fetch location"

# Get unique lat/lon pairs
unique_coords = data[['LAT', 'LON']].drop_duplicates()

# Initialize a dictionary to store location names
location_dict = {}

# Iterate over unique coordinates and get location names
for idx, row in tqdm(unique_coords.iterrows(), total=unique_coords.shape[0]):
    lat, lon = row['LAT'], row['LON']
    location_name = get_location_name(lat, lon)
    location_dict[(lat, lon)] = location_name

# Convert location_dict to a DataFrame and save it
location_df = pd.DataFrame(list(location_dict.items()), columns=['Coordinates', 'Location'])
location_df[['LAT', 'LON']] = pd.DataFrame(location_df['Coordinates'].tolist(), index=location_df.index)
location_df.drop(columns=['Coordinates'], inplace=True)
location_df.to_csv('location_cache.csv', index=False)

print("Location names cached successfully.")
