#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

# Example property data
property_data = {
    "bhk": 2,
    "size": 1100,
    "bathroom": 2,
    "area_type": "super_area",
    "city": "mumbai",
    "furnishing_status": "semi-furnished",
    "tenant_preferred": "bachelors/family",
    "point_of_contact": "contact_owner"
}

print("Sending request to:", url)
print("Property data:", property_data)
print()

response = requests.post(url, json=property_data)
result = response.json()

print("Response:")
print(f"  Predicted rent: {result['predicted_rent']:,.2f}")
print(f"  Predicted rent (log): {result['predicted_rent_log']:.4f}")