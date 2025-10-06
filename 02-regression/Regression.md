# Machine Learning for Regression

## Useful functions

- `sns.histplot(df, bins=50)` – plots a histogram to visualize the distribution of values.
- `np.log()` – applies the natural logarithm, useful for reducing skewness.
- `np.log1p()` – computes `log(x + 1)`, safe for zero values.
- `df.isnull().sum()` – counts missing values in each column.
- `del df['col']` – deletes a selected column from the dataframe.
- `df.copy()` – creates a copy of the dataframe to work safely on data.
