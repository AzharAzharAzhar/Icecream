# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:08:27 2023

@author: Indoskill
"""


import pickle as pkl
import numpy as np
import streamlit as st

filename = "Salesdata_trainedmodel.sav"
load_model = pkl.load(open(filename, 'rb'))

st.title('Ice Cream Sales Revenue Prediction')
input_data = st.number_input("Enter the Temperarture Value: ", max_value=100, value=0)
#result = st.write("The prediction is")

def predictions(data):
    data_array = np.asarray(data).reshape(-1,1)
    pred = load_model.predict(data_array)
    return pred[0]

def main():
    #input_data = 12
    #print(predictions(input_data))
    if st.button("Predict"):
        pred = predictions(input_data)
        st.write("The prediction is "+ str(pred))
if __name__ == "__main__":
    main()
    
