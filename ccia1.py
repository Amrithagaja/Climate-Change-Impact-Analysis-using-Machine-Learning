# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 01:46:19 2024

@author: Jai
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Climate Change Impact",page_icon="C://Users//Jai//Downloads//ccia.jpg")
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
# Run the app
st.title("Climate Change Impact Analysis")
st.write("Select a parameter and click on the map to view predictions.")
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

# Event handling for map click or hover using st.query_params
query_params = st.experimental_get_query_params()
if 'lat' in query_params and 'lon' in query_params:
    lat = float(query_params['lat'][0])
    lon = float(query_params['lon'][0])

    # Find closest data point to clicked location
    closest_data = melted_data.loc[((melted_data['LAT'] - lat).abs() + 
                                    (melted_data['LON'] - lon).abs()).idxmin()]

    # Extract information for display
    predicted_month = closest_data['Month']
    predicted_value = closest_data['Value']

    # Display the prediction information
    prediction_text = f"Predicted {parameter} at {lat}, {lon} for {predicted_month}: {predicted_value}"
    st.write(prediction_text)



