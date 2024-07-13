# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:34:18 2024

@author: Jai
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
from tqdm import tqdm

# Load your data
data = pd.read_csv('C://Users//Jai//Downloads//POWER_Regional_Monthly_1984_2022.csv')

# Ensure LAT and LON are numeric
data['LAT'] = pd.to_numeric(data['LAT'], errors='coerce')
data['LON'] = pd.to_numeric(data['LON'], errors='coerce')

# Initialize geocoder with a longer timeout
geolocator = Nominatim(user_agent="geoapiExercises", timeout=10)

# Function to get location name from coordinates with retry mechanism
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
