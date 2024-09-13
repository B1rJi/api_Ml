#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:44:52 2024

@author: ankitpandey
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import numpy as np

app = FastAPI()

# Define input schema for the API using Pydantic's BaseModel
class ModelInput(BaseModel):
    pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the saved model and scaler
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))  # Load the saved scaler

@app.post('/diabetes_prediction')
def diabetes_predict(input_parameters: ModelInput):
    # Convert input parameters to JSON, then load as a dictionary
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    # Extract the input values
    preg = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']

    # Prepare the input list and scale it
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    input_array = np.array(input_list).reshape(1, -1)  # Reshape for scaling

    # Apply the same scaling transformation as during model training
    scaled_input = scaler.transform(input_array)

    # Predict the outcome using the loaded model
    prediction = diabetes_model.predict(scaled_input)

    # Return the prediction result
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
