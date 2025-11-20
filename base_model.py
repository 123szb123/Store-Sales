# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv (3.13.3)
#     language: python
#     name: python3
# ---

# %%
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from scipy.signal import periodogram
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier # For modeling trends and seasonal features

# %%
# Datasets
train = pd.read_csv('data/train.csv')
stores = pd.read_csv('data/stores.csv')

print('TRAINING DATA \n')
print(train.info())
print('\n')
print('STORES METADATA \n')
print(stores.info())

# %%
# convert date strings to datetime objects
train['date'] = pd.to_datetime(train['date'])

# %%
df = pd.merge(
    train,
    stores,
    on = 'store_nbr',
    how = 'left'
)

df.info()

# %%
# we drop id feature but only if it exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# %%
df = df.sort_values('date')

df['dow'] = df['date'].dt.dayofweek # 0=mon 1=tue 2=wed 3=thu 4=fri 5=sat 6=sun because monday ≠ saturday in sales
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['is_weekend'] = (df['dow'] >= 5).astype(int) # because weekends are spikes
df['time_idx'] = (df['date'] - df['date'].min()).dt.days  # linear time trend

df = df.sort_values(['store_nbr','family','date']) # add lag features

df['lag7'] = df.groupby(['store_nbr','family'])['sales'].shift(7) # 1 week lag to capture weekly trends
df['roll7'] = ( # smooth it out by finding average behavior over past week
    df.groupby(['store_nbr','family'])['sales']
      .shift(1)
      .rolling(7)
      .mean()
) # 7 day rolling mean gets lagged by 1 day

# now drop rows where lag features are null or nan (the first ~7 days per store/family)
df_model = df.dropna(subset=['lag7','roll7']).copy()

# okay so apparently this works because previously the mmodel had no clue that demand is always high on specific dates (like december 23 or 24)
# so it was underpredicting peaks and overpredicting quiet days. this essentially teaches the model the calendar

# %%
categorical_features = [
    'store_nbr',
    'family',
    'city',
    'state',
    'type',
    'cluster',
    'dow',
    'month'
]

numeric_features = [
    'onpromotion',  # onpromotion was previously treated ascategorical so changed it
    'lag7',
    'roll7', 
    'is_weekend',
    'time_idx' # linear time trend which is from 0 to the length of the dataset
]


# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
) # preprocess numeric features with standard scaler and categorical with one hot encoder

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
]) # what this does is first preprocess the data then eventually fit the model

# %%
X = df.drop('sales',axis=1)
y = df['sales']

from sklearn.model_selection import train_test_split # split data into training and cross-validation sets

X_train, X_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3, random_state=57)

# %%
df_model = df_model.sort_values('date') # ensure chronological order

feature_cols = categorical_features + numeric_features # columns we want the model to see

X_all = df_model[feature_cols]
y_all = df_model['sales']
dates = df_model['date']

cutoff_date = dates.quantile(0.8) # 80 percent cutoff date for training vs validation (apparently 80 is a good number)

# so if we had 100 days of data, the first 80 days would be training and last 20 days would be used for validation

train_mask = dates <= cutoff_date # training data is dates on or before cutoff date so the first ~80%
cv_mask    = dates > cutoff_date # cross-validation data is dates after cutoff date so the last ~20%

X_train = X_all[train_mask] # these are training features
y_train = y_all[train_mask] # training targets 
# these are passed into fit function of model_pipeline

X_cv = X_all[cv_mask] # cross-validation features which we will pass into predict function
y_cv = y_all[cv_mask] # now we got the cross-validation targets to pass into rsmle function

# %%
model_pipeline.fit(X_train,y_train)
predictions = model_pipeline.predict(X_cv)
predictions = np.maximum(predictions,0)

# %%
predictions[predictions < 0]

# %%
from sklearn.metrics import root_mean_squared_log_error

rmsle = root_mean_squared_log_error(y_cv,predictions)
print(f'RMSLE: {rmsle: .4f}')

# %% [markdown]
# ---

# %%

tot_sales = df.groupby('date')['sales'].sum().reset_index()
tot_sales = tot_sales.set_index('date').asfreq('D').fillna(0)

tot_sales = tot_sales.reset_index()  
tot_sales['date'] = pd.to_datetime(tot_sales['date'])
tot_sales = tot_sales.set_index('date').sort_index()

dp = DeterministicProcess(
    index = tot_sales.index,
    constant=True,
    order=1, # Linear trend
    seasonal=True,
    period=365, # Yearly seasonality
    drop=True
)

# Training features
X_train = dp.in_sample()
y = tot_sales['sales']

# Fit the linear model
model = LinearRegression()
model.fit(X_train,y)

# Make predictions on training data, for evaluation
y_pred = model.predict(X_train)

y_pred = np.maximum(y_pred, 0) # forces no negative predictions

# Calculate training error (root mean squared error)
rmsle = np.sqrt(mean_squared_log_error(y,y_pred))
print(f'Training RMSLE: {rmsle:.2f}')

# Out-of-sample predictions
forecast_steps = 16
X_forecast = dp.out_of_sample(steps=forecast_steps)
forecast = model.predict(X_forecast)

# forecast dataframe
last_date = tot_sales.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=forecast_steps, 
                                freq='D')

forecast_df = pd.DataFrame({
    'date' : forecast_dates,
    'forecast': forecast
})

print("\nForecast:")
print(forecast_df)
