# 02-regression -  Car price prediction project

##Project plan:

* Prepare data and Exploratory data analysis (EDA)
* Use linear regression for predicting price
* Understanding the internals of linear regression 
* Evaluating the model with RMSE
* Feature engineering  
* Regularization 
* Using the model 

## Useful functions

- `sns.histplot(df, bins=50)` – plots a histogram to visualize the distribution of values.
- `np.log()` – applies the natural logarithm, useful for reducing skewness.
- `np.log1p()` – computes `log(x + 1)`, safe for zero values.
- `df.isnull().sum()` – counts missing values in each column.
- `del df['col']` – deletes a selected column from the dataframe.
- `df.copy()` – creates a copy of the dataframe to work safely on data.
