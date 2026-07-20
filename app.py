import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load and train model
os.chdir(r'C:\Users\admin\OneDrive\Desktop\CropYield-Project')
df = pd.read_csv('Crop Yeild Data.csv')

# Data Cleaning
df = df[df['Crop_Year'] != 2020]
df = df.drop(columns=['Production'])
df['Temp_Range'] = df['Max_Temperature'] - df['Min_Temperature']

# Outlier handling
Q1 = df['Yield'].quantile(0.25)
Q3 = df['Yield'].quantile(0.75)
IQR = Q3 - Q1
df['Yield'] = df['Yield'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)

# Label Encoding
le_crop = LabelEncoder()
le_season = LabelEncoder()
le_state = LabelEncoder()

df['Crop'] = le_crop.fit_transform(df['Crop'])
df['Season'] = le_season.fit_transform(df['Season'])
df['State'] = le_state.fit_transform(df['State'])

# Train model
X = df.drop(columns=['Yield'])
y = df['Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# App
st.title('🌾 Crop Yield Predictor — India')
st.write('Enter crop and weather details to predict yield!')

col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox('Crop', le_crop.classes_)
    season = st.selectbox('Season', le_season.classes_)
    state = st.selectbox('State', le_state.classes_)
    crop_year = st.number_input('Crop Year', min_value=1997, max_value=2024, value=2020)
    area = st.number_input('Area (Hectares)', min_value=0.0, value=1000.0)
    annual_rainfall = st.number_input('Annual Rainfall (mm)', min_value=0.0, value=1000.0)

with col2:
    fertilizer = st.number_input('Fertilizer (kg)', min_value=0.0, value=100000.0)
    pesticide = st.number_input('Pesticide (kg)', min_value=0.0, value=1000.0)
    avg_temp = st.number_input('Avg Temperature (°C)', min_value=-10.0, value=25.0)
    max_temp = st.number_input('Max Temperature (°C)', min_value=-10.0, value=35.0)
    min_temp = st.number_input('Min Temperature (°C)', min_value=-10.0, value=15.0)

if st.button('Predict Yield'):
    temp_range = max_temp - min_temp
    crop_encoded = le_crop.transform([crop])[0]
    season_encoded = le_season.transform([season])[0]
    state_encoded = le_state.transform([state])[0]

    input_data = np.array([[crop_encoded, crop_year, season_encoded, state_encoded,
                            area, annual_rainfall, fertilizer, pesticide,
                            avg_temp, max_temp, min_temp, temp_range]])

    prediction = model.predict(input_data)[0]

    st.markdown(f'### 🌾 Predicted Yield: {round(prediction, 2)} Tonnes/Hectare')

    if prediction < 1:
        st.warning('Low yield — consider changing crop or improving conditions!')
    elif prediction < 3:
        st.success('Moderate yield — good conditions!')
    else:
        st.success('High yield — excellent conditions!')