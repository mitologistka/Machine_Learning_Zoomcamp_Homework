import pandas as pd
import numpy as np
import kagglehub

from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
file_path = "House_Rent_Dataset.csv"
df_base = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "iamsouravbanerjee/house-rent-prediction-dataset",
    file_path
)

# Clean column names
df_base.columns = df_base.columns.str.lower().str.replace(' ', '_')

# Drop unnecessary columns
df = df_base.drop(columns=["floor", 'area_locality', "posted_on"])

# Lowercase string columns
strings = list(df.dtypes[df.dtypes == 'object'].index)
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_') 

# --- Split data ---
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.rent.values)
y_val = np.log1p(df_val.rent.values)
y_test = np.log1p(df_test.rent.values)

for d in [df_train, df_val, df_test]:
    del d['rent']

# --- Features ---
numerical = ['bhk', 'size', 'bathroom']
categorical = ['area_type', 'city', 'furnishing_status', 'tenant_preferred', 'point_of_contact']

# --- DictVectorizer ---
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

# --- Linear Regression ---
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_val)

# --- Ridge ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train_scaled, y_train)
y_pred_ridge = model_ridge.predict(X_val_scaled)

# --- Decision Tree ---
model_dt = DecisionTreeRegressor(random_state=1, max_depth=10, min_samples_split=2, min_samples_leaf=5, max_features=None)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_val)

# Decision Tree Grid Search
param_grid_dt = {
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ['sqrt', None]
}
grid_search_dt = GridSearchCV(DecisionTreeRegressor(random_state=1),
    param_grid_dt,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
print("Decision Tree Best params:", grid_search_dt.best_params_)
print("Decision Tree Best CV RMSE:", -grid_search_dt.best_score_)

# --- Random Forest ---
model_rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=1,
    max_features=None, random_state=1, n_jobs=-1)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_val)

# Random Forest Grid Search
param_grid_rf = {
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ['sqrt', None]
}
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=1),
    param_grid_rf,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
print("Random Forest Best params:", grid_search_rf.best_params_)
print("Random Forest Best CV RMSE:", -grid_search_rf.best_score_)

# --- Compare models ---
results = {
    'Linear Regression': {'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_lr)),
    'R2': r2_score(y_val, y_pred_lr)},
    'Ridge': {'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_ridge)),
    'R2': r2_score(y_val, y_pred_ridge)},
    'Decision Tree': {'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_dt)),
    'R2': r2_score(y_val, y_pred_dt)},
    'Random Forest': {'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_rf)),
    'R2': r2_score(y_val, y_pred_rf)}
}
df_results = pd.DataFrame(results).T.sort_values(by='RMSE')
print(df_results)

# --- Plot results ---
df_results.plot(kind='bar', y=['RMSE','R2'], subplots=True, figsize=(10,6), legend=True)
plt.tight_layout()
plt.show()
