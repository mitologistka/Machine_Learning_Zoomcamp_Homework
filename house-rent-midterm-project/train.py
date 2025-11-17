#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Parameters
output_file = 'model.bin'
random_state = 1

print("Loading data from Kaggle...")

# Load data
file_path = "House_Rent_Dataset.csv"
df_base = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "iamsouravbanerjee/house-rent-prediction-dataset",
    file_path
)

print(f"Dataset loaded: {df_base.shape}")

# Data preparation
print("Preparing data...")

df_base.columns = df_base.columns.str.lower().str.replace(' ', '_')
df = df_base.drop(columns=["floor", 'area_locality', "posted_on"])

# Process string columns
strings = list(df.dtypes[df.dtypes == 'object'].index)
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Split data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=random_state)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Target variable (log transformation)
y_train = np.log1p(df_train.rent.values)
y_val = np.log1p(df_val.rent.values)
y_test = np.log1p(df_test.rent.values)

del df_train['rent']
del df_val['rent']
del df_test['rent']

print(f"Train size: {len(df_train)}")
print(f"Validation size: {len(df_val)}")
print(f"Test size: {len(df_test)}")

# Define features
numerical = ['bhk', 'size', 'bathroom']
categorical = [
    'area_type',
    'city',
    'furnishing_status',
    'tenant_preferred',
    'point_of_contact'
]

# Vectorize features
print("Vectorizing features...")

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

# Train the final model on full training data
print("Training the final model...")

# Use full training data (train + validation)
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = np.log1p(df_full_train.rent.values)
del df_full_train['rent']

full_train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

model = LinearRegression()
model.fit(X_full_train, y_full_train)

# Evaluate on test set
test_dict = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dict)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")

# Save the model
print(f"Saving model to {output_file}...")

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Model saved successfully!")