# Climate-Change-Impact-Analysis
**Climate Change Impact ML App**

This Streamlit web application analyzes climate change data and provides machine learning predictions based on geographical locations.

**Overview**

The app visualizes climate data on a map and allows users to select parameters for analysis. It integrates machine learning to predict climate impacts at specified locations and months.

**Flowchart**
![ML](https://github.com/user-attachments/assets/57eba67f-a0a5-4904-bd4d-8467fce23fc8)




**Features**

Interactive Map Visualization: Displays climate data points on an interactive map using Plotly and Mapbox.
Machine Learning Prediction: Utilizes a linear regression model to predict climate values based on selected parameters, location, and month.
User Interface: Offers a sidebar for parameter selection and dynamic updates based on user interactions with the map.

**Requirements**

Ensure you have Python 3.x installed. Install required Python packages using:

pip install -r requirements.txt

Required packages include Streamlit, Plotly, pandas, numpy, scikit-learn, geopy, and tqdm.

**Installation**

Clone the repository:
git clone <repository-url>
cd <repository-directory>

**Install dependencies:**

pip install -r requirements.txt

**Run the Streamlit app:**

streamlit run app.py

**Usage**

Parameter Selection: Use the sidebar to select a climate parameter for analysis.
Map Interaction: Click or hover over map locations to view predicted climate values for specific months.
Model Evaluation: Evaluates the predictive model's Mean Squared Error (MSE) based on training and test data splits.

**Data Sources**

The climate data (POWER_Regional_Monthly_1984_2022.csv) includes parameters such as temperature, precipitation, and more, across various geographical locations globally.

**Notes**

Location Caching: Geographical location names are cached locally (location_cache.csv) for faster retrieval during app usage.
Contributing
Contributions are welcome. Please fork the repository and submit pull requests.

