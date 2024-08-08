## Machine Learning model Deployment using Streamlit

import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 


pickle_in_oil = open('model_oil.pkl', 'rb')
model_oil = pickle.load(pickle_in_oil)

pickle_in_gas = open('model_gas.pkl', 'rb')
model_gas = pickle.load(pickle_in_gas)

pickle_in_water = open('model_water.pkl', 'rb')
model_water = pickle.load(pickle_in_water)

st.set_page_config(page_title="Well Forecast App")
def predictions(Date,Downhole_Pressure,Downhole_Temperature,
                Average_Tubing_Pressure,Annulus_Pressure,
                Average_Wellhead_Pressure,Choke_Size,WellBore_Name):
    
    # Preprocessing User input - Date
    Date = pd.to_datetime(Date)
    day_of_week=Date.day_of_week
    year = Date.year
    yearly_quarter=Date.quarter
    day_of_year=Date.dayofyear
    month_of_year= Date.month

    # Preprocessing User input - Categorical Encoding

    if WellBore_Name == "001_F_12":
        bin_list = [0,0,1,0,0]    
    elif WellBore_Name == '001_F_14':
        bin_list = [0,0,0,1,0]
    elif WellBore_Name =='001_F_11':
        bin_list = [0,1,0,0,0]
    elif WellBore_Name == '001_F_15 D':
        bin_list = [0,0,0,0,1]
    elif WellBore_Name == '001_F_1 C':
        bin_list = [1,0,0,0,0]
    # Engineered Features
    pressure_differential = Downhole_Pressure - Average_Wellhead_Pressure
    pressure_differential_annulus = Average_Tubing_Pressure - Annulus_Pressure
    P_T_Ratio = Downhole_Pressure/Downhole_Temperature
    
    # Oil Production Predictions 
    oil_user_input = [Downhole_Pressure, Downhole_Temperature,Average_Tubing_Pressure, 
                      Annulus_Pressure ,Average_Wellhead_Pressure, Choke_Size, 
                      day_of_week, year,yearly_quarter,day_of_year, month_of_year, 
                      pressure_differential,pressure_differential_annulus, P_T_Ratio]
    i = 0
    while i < len(bin_list):
        oil_user_input.insert(6+i, bin_list[i])
        i = i + 1
    oil_predictions = model_oil.predict([oil_user_input])                                 
    oil_predictions = max(0, oil_predictions)

    # Gas Production Predictions
    gas_user_input = [Downhole_Pressure, Downhole_Temperature,Average_Tubing_Pressure, 
                      Annulus_Pressure ,Average_Wellhead_Pressure, Choke_Size, 
                      day_of_week, year, yearly_quarter,day_of_year, month_of_year,
                      pressure_differential, pressure_differential_annulus] 
    i = 0
    while i < len(bin_list):
        gas_user_input.insert(6+i, bin_list[i])
        i = i + 1
    gas_predictions = model_gas.predict([gas_user_input])
    gas_predictions = max(0, gas_predictions)

    #Water Production Predictions
    water_user_input = [Downhole_Pressure, Downhole_Temperature, Average_Tubing_Pressure, 
                      Annulus_Pressure ,Average_Wellhead_Pressure, Choke_Size, 
                      day_of_week, year, pressure_differential, pressure_differential_annulus
                      ]
    i = 0
    while i < len(bin_list):
        water_user_input.insert(6+i, bin_list[i])
        i = i + 1
    water_predictions = model_water.predict([water_user_input])
    water_predictions = max(0, water_predictions)
    
    return oil_predictions, gas_predictions, water_predictions
    

    
def main():
    st.title("Well Production Forecasting using Machine Learning")
    st.write("This web app utilizes machine learning techniques to accurately forecast production rates of oil, gas, and water from a production well. The model was trained on data from a conceptual oil field operated between 2008 and 2015.")
    st.write("<i>Fill in the parameters below to generate a well production estimate:</i>",unsafe_allow_html=True)
    
    Date = st.date_input("Forecast Date")
    WellBore_Name = st.selectbox("Wellbore Name", ['001_F_12', '001_F_14', '001_F_11', '001_F_15 D', '001_F_1 C'],
                                 help="Wellbore IDs as provided in training data")
    Downhole_Temperature = st.slider('Downhole Temperature(kelvin)', min_value=273, max_value=400)
    Downhole_Pressure = st.slider('Downhole Pressure(psi)', min_value=0, max_value=6000)  

    Average_Tubing_Pressure = st.slider('Average Tubing Pressure(psi)', min_value=0, max_value=5000)
    Annulus_Pressure = st.slider('Annulus Pressure(psi)', min_value=0, max_value=450)
    Average_Wellhead_Pressure = st.slider('Average Wellhead Pressure(psi)', min_value=0, max_value=2000)
    Choke_Size = st.slider('Choke Size(/64)', min_value=0, max_value=100)
    
    if st.button("Production Forecast"):
        oil_prediction,gas_prediction,water_prediction=predictions(Date,Downhole_Pressure,Downhole_Temperature,
                                                                Average_Tubing_Pressure,Annulus_Pressure,
                                                                Average_Wellhead_Pressure,Choke_Size,WellBore_Name)
    
        st.success(f"Oil Production Forecast: {"{:,}".format(int(oil_prediction))} stb/day")
        st.success(f"Gas Production Forecast: {"{:,}".format(int(gas_prediction))} scf/day")
        st.success(f"Water Production Forecast: {"{:,}".format(int(water_prediction))} stb/day")
if __name__=='__main__': 
    main()



        
    
    
    
    
    



        
    
