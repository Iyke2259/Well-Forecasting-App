## Machine Learning model Deployment using Streamlit

import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 

pickle_in_gas = open('model_gas.pkl', 'rb')
model_gas = pickle.load(pickle_in_gas)

pickle_in_water = open('model_water.pkl', 'rb')
model_water = pickle.load(pickle_in_water)

pickle_in_oil = open('model_oil.pkl', 'rb')
model_oil = pickle.load(pickle_in_oil)



st.set_page_config(page_title="Well Forecast App")
def predictions(Date,Downhole_Pressure,Downhole_Temperature,Average_Tubing_Pressure,Annulus_Pressure,Average_Wellhead_Pressure,Choke_Size,WellBore_Name):
    
    # Preprocessing User input - Date
    Date = pd.to_datetime(Date)
    day_of_week=Date.day_of_week
    year = Date.year
    yearly_quarter=Date.quarter
    day_of_year=Date.dayofyear
    month_of_year= Date.month

    # Preprocessing User input - Categorical Encoding

    if WellBore_Name == "001_F_12":
        WellBore_Name_001_F_1_C = 0
        WellBore_Name_001_F_11 = 0
        WellBore_Name_001_F_12 = 1 
        WellBore_Name_001_F_14= 0
        WellBore_Name_001_F_15_D =0
    elif WellBore_Name == '001_F_14':
        WellBore_Name_001_F_1_C = 0
        WellBore_Name_001_F_11 = 0
        WellBore_Name_001_F_12 = 0 
        WellBore_Name_001_F_14= 1
        WellBore_Name_001_F_15_D =0
    elif WellBore_Name =='001_F_11':
        WellBore_Name_001_F_1_C = 0
        WellBore_Name_001_F_11 = 1
        WellBore_Name_001_F_12 = 0
        WellBore_Name_001_F_14= 0
        WellBore_Name_001_F_15_D =0
    elif WellBore_Name == '001_F_15 D':
        WellBore_Name_001_F_1_C = 0
        WellBore_Name_001_F_11 = 0
        WellBore_Name_001_F_12 = 0 
        WellBore_Name_001_F_14= 0
        WellBore_Name_001_F_15_D =1
    elif WellBore_Name == '001_F_1 C':
        WellBore_Name_001_F_1_C = 1
        WellBore_Name_001_F_11 = 0
        WellBore_Name_001_F_12 = 1 
        WellBore_Name_001_F_14= 0
        WellBore_Name_001_F_15_D =0
    # Engineered Features
    pressure_differential = Downhole_Pressure - Average_Wellhead_Pressure
    pressure_differential_annulus = Average_Tubing_Pressure - Annulus_Pressure
    P_T_Ratio = Downhole_Pressure/Downhole_Temperature
    # Predictions 
    oil_predictions = model_oil.predict([[Downhole_Pressure, Downhole_Temperature,Average_Tubing_Pressure, 
                                          Annulus_Pressure ,Average_Wellhead_Pressure, Choke_Size, 
                                          WellBore_Name_001_F_1_C, WellBore_Name_001_F_11, WellBore_Name_001_F_12,
                                          WellBore_Name_001_F_14, WellBore_Name_001_F_15_D, day_of_week, year, 
                                          yearly_quarter,day_of_year, month_of_year, pressure_differential,
                                          pressure_differential_annulus, P_T_Ratio
                                         ]])
    oil_predictions = max(0, oil_predictions)
        
    gas_predictions = model_gas.predict([[Downhole_Pressure, Downhole_Temperature,Average_Tubing_Pressure, 
                                          Annulus_Pressure ,Average_Wellhead_Pressure, Choke_Size, 
                                          WellBore_Name_001_F_1_C, WellBore_Name_001_F_11, WellBore_Name_001_F_12,
                                          WellBore_Name_001_F_14, WellBore_Name_001_F_15_D, day_of_week, year, 
                                          yearly_quarter,day_of_year, month_of_year, pressure_differential,
                                          pressure_differential_annulus]])
    gas_predictions = max(0, gas_predictions)
       
    water_predictions = model_water.predict([[Downhole_Pressure, Downhole_Temperature, Average_Tubing_Pressure, 
                                              Annulus_Pressure ,Average_Wellhead_Pressure, Choke_Size, 
                                              WellBore_Name_001_F_1_C, WellBore_Name_001_F_11, WellBore_Name_001_F_12,
                                              WellBore_Name_001_F_14, WellBore_Name_001_F_15_D, day_of_week, year, 
                                              pressure_differential, pressure_differential_annulus]])
    water_predictions = max(0, water_predictions)
    
    return oil_predictions, gas_predictions, water_predictions
    

    
def main():
    st.title("Well Production Forecasting using Machine Learning")
    #st.sidebar.header(" Web App")
    st.write("This web app utilizes machine learning techniques to accurately forecast the production rates for the oil, gas, and water. The ML model was trained on well production data from a conceptual oil field between 2008 and 2015.")
    st.write("<i>Fill in the appropriate parameters below to generate a well production estimate:</i>",unsafe_allow_html=True)
    
      
    Date = st.date_input("Forecast Date",help="ML model was trained on production data between 2008 and 2015")
    WellBore_Name = st.selectbox("Wellbore Name", ['001_F_12', '001_F_14', '001_F_11', '001_F_15 D', '001_F_1 C'],
                                 help="Wellbore IDs as provided in training data")
    Downhole_Temperature = st.slider('Downhole Temperature(kelvin)', min_value=273, max_value=400)
    Downhole_Pressure = st.slider('Downhole Pressure(psi)', min_value=0, max_value=6000)  

    Average_Tubing_Pressure = st.slider('Average Tubing Pressure(psi)', min_value=0, max_value=5000)
    Annulus_Pressure = st.slider('Annulus Pressure(psi)', min_value=0, max_value=450)
    Average_Wellhead_Pressure = st.slider('Average Wellhead Pressure(psi)', min_value=0, max_value=2000)
    Choke_Size = st.slider('Choke Size(/64)', min_value=0, max_value=100)

    # add element on the right side
    # User Input
   
    
    if st.button("Production Forecast"):
        #st.progress(100,text= "Operation in progress. Please wait.")
        oil_prediction, gas_prediction, water_prediction  = predictions(Date,Downhole_Pressure,Downhole_Temperature,Average_Tubing_Pressure,
                    Annulus_Pressure,Average_Wellhead_Pressure,Choke_Size,WellBore_Name)
    
        st.success(f"Oil Production Forecast: {"{:,}".format(int(oil_prediction))} stb/day")
        st.success(f"Gas Production Forecast: {"{:,}".format(int(gas_prediction))} scf/day")
        st.success(f"Water Production Forecast: {"{:,}".format(int(water_prediction))} stb/day")
if __name__=='__main__': 
    main()



        
    
    
    
    
    



        
    
