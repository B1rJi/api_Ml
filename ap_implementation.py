#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:02:42 2024

@author: ankitpandey
"""


import json
import requests


url = 'http://127.0.0.1:8080/diabetes_prediction'

input_data_for_model = {
    
    'pregnancies' : 5,
    'Glucose' : 166,
    'BloodPressure' : 72,
    'SkinThickness' : 19,
    'Insulin' : 175,
    'BMI' : 25.8,
    'DiabetesPedigreeFunction' : 0.587,
    'Age' : 51
    
    }

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.text)