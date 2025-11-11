import pickle
import pandas as pd
import numpy as np

# Sklearn for splitting data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # for cross-validation

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score # for model evaluation

# parameters
C = 1.0
# cross-validation with k=5
n_splits = 5
output_file = f'model_C={C}.bin'

df = pd.read_csv('data-week-3.csv')

# change column names to lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')

# make list of names of categorical columns
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

# lowercase and replace spaces with underscores in categorical columns
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# convert totalcharges to numeric, coerce errors to NaN
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

# convert target column to binary
df.churn = (df.churn == 'yes').astype(int)

# Split data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# C is the regularization parameter
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    # convert dicts to feature matrix
    dv = DictVectorizer(sparse=False)
    # learn the feature mapping and transform the data to binary matrix
    X_train = dv.fit_transform(dicts)

    # train the model
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    # convert dataframe to dicts
    dicts = df[categorical + numerical].to_dict(orient='records')

    # transform dicts to feature matrix
    X = dv.transform(dicts)
    # predict probabilities
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


print(f'Doing validation with C={C}')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    # split the data using kfold indexes
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    # target values 
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    # train the model and make predictions
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)
    
    # how good are the predictions?
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# train the final model on the full training data
print('training the final model')
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
print(f'auc: {auc}')

# Save the model
output_file = f'model_C={C}.bin' # save the model to a binary file

print(f'Saving the model to {output_file}')

