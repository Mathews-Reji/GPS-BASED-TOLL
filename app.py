import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, time
from predictions import predict

# Debug: Check model loading
try:
    model = joblib.load("06_07_lgbm_model.sav")
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title('GPS BASED TOLL COLLECTION')
st.markdown('A test model created for calculating the fees according to the distance travelled.')

st.header("Journey Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.text("Starting Point")
    start_x = st.text_input("Latitude_start")
    start_y = st.text_input("Longitude_start")
    st.text('')
    st.text("End Points")
    end_x = st.text_input("Latitude_end")
    end_y = st.text_input("Longitude_end")

with col2:
    st.text('Time Details')
    start_hour = st.text_input("Enter Start Hour")
    start_minute = st.text_input("Enter Start Minute")
    end_minute = st.text_input("Enter End Minute")
    end_second = st.text_input("Enter End Seconds")

with col3:
    st.text("Other Details")
    distance = st.text_input("Enter The Distance")
    vehicle_id = st.text_input("Enter The Vehicle ID")
    average_speed = st.number_input('Enter a Avg Speed')

st.text('')
if st.button("Calculate Fee"):
    try:
        # Convert inputs to appropriate types
        start_x = float(start_x)
        start_y = float(start_y)
        end_x = float(end_x)
        end_y = float(end_y)
        start_hour = int(start_hour)
        start_minute = int(start_minute)
        end_minute = int(end_minute)
        end_second = int(end_second)
        distance = float(distance)
        average_speed = float(average_speed)

        # Prepare the input data for prediction
        input_data = {
            'start_hour': start_hour,
            'start_minute': start_minute,
            'end_minute': end_minute,
            'end_second': end_second,
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'distance': distance,
            'average_speed': average_speed
        }

        # Initialize vehicle ID encoding
        vehicle_ids = ['H', 'M', 'S', 'T']
        for vid in vehicle_ids:
            input_data[f'vehicle_id_{vid}'] = 1 if vehicle_id == vid else 0

        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Debug: Print input DataFrame
        st.write("Input Data for Prediction (DataFrame):", input_df)

        # Extract vehicle ID encoded columns
        vehicle_id_H = input_df['vehicle_id_H'].values[0]
        vehicle_id_M = input_df['vehicle_id_M'].values[0]
        vehicle_id_S = input_df['vehicle_id_S'].values[0]
        vehicle_id_T = input_df['vehicle_id_T'].values[0]

        # Predict the fee using the external predict function
        result = predict(np.array([[start_hour, start_minute, end_minute, end_second, start_x, start_y, end_x, end_y, distance, average_speed, vehicle_id_H, vehicle_id_M, vehicle_id_S, vehicle_id_T]]))
        
        # Display the result
        st.success(f"The calculated toll fee is: {result[0]:.2f}")

    except ValueError:
        st.error("Please enter valid input values.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
