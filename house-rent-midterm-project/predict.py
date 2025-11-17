#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the model
model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('rent-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict house rent based on property features.
    
    Expected JSON input:
    {
        "bhk": 2,
        "size": 1100,
        "bathroom": 2,
        "area_type": "super_area",
        "city": "mumbai",
        "furnishing_status": "semi-furnished",
        "tenant_preferred": "bachelors/family",
        "point_of_contact": "contact_owner"
    }
    """
    property_data = request.get_json()
    
    # Transform the input
    X = dv.transform([property_data])
    
    # Make prediction (log scale)
    y_pred_log = model.predict(X)[0]
    
    # Convert back to original scale
    y_pred = np.expm1(y_pred_log)
    
    result = {
        'predicted_rent': float(y_pred),
        'predicted_rent_log': float(y_pred_log)
    }
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

    