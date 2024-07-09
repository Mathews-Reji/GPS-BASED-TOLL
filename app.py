import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, time
from predictions import predict

# Load the trained model
model = joblib.load("06_07_lgbm_model.sav")
#recommit

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
    start_hour=st.text_input("Enter Start Hour")
    start_minute=st.text_input("Enter Start Minute")
    end_minute=st.text_input("Enter End Minute")
    end_second=st.text_input("Enter End Seconds")
    #end_msecond=st.text_input("Enter Milliseconds")
    

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
        start_hour=int(start_hour)
        start_minute=int(start_minute)
        end_minute=int(end_minute)
        end_second=int(end_second)
        #end_msecond=int(end_msecond)
        distance = float(distance)
        average_speed = float(average_speed)

        # Select relevant features (inputs) and target (output)
        base_features = ['start_hour', 'start_minute','end_minute','end_second','start_x','start_y','end_x','end_y','distance','average_speed' ]
        

        # Convert categorical vehicle_id feature to numerical using one-hot encoding
        df = pd.get_dummies(df, columns=['vehicle_id'])

        # Update features list to include the one-hot encoded vehicle_id columns
        encoded_vehicle_id_columns = [col for col in df.columns if 'vehicle_id' in col]
        features = base_features + encoded_vehicle_id_columns


        # Predict the fee
        result = model.predict(features)
        
        # Display the result
        st.success(f"The calculated toll fee is: {result[0]:.2f}")

        
    except ValueError:
        st.error("Please enter valid input values.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
