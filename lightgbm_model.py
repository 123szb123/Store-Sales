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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path # use relative paths cleanly (data/train.csv, etc.)

from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import joblib # for saving/loading models (idrk how this work tbh)

from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

# part 1 - loads data

DATA_DIR = Path("data")

train = pd.read_csv(DATA_DIR / "train.csv") 
test = pd.read_csv(DATA_DIR / "test.csv")
stores = pd.read_csv(DATA_DIR / "stores.csv")
oil = pd.read_csv(DATA_DIR / "oil.csv")
holidays = pd.read_csv(DATA_DIR / "holidays_events.csv")
transactions = pd.read_csv(DATA_DIR / "transactions.csv")

print("TRAIN INFO\n")
print(train.info()) # prints summary of train dataframe
print("\nSTORES INFO\n")
print(stores.info()) # prints summary of stores dataframe

# converts date columns to datetime
for df in [train, test, oil, holidays, transactions]: # iterates through all dataframes
    df["date"] = pd.to_datetime(df["date"]) # converts the date column to datetime format

# part 2 - merges data

# now we merge train with external data to get a richer feature table
train_full = train.merge(oil, on="date", how="left")
train_full = train_full.merge(stores, on="store_nbr", how="left")

holidays_simple = holidays[["date", "type", "locale", "locale_name", "transferred"]] 
# simplifies holidays dataframe by selecting only relevant columns
train_full = train_full.merge(holidays_simple, on="date", how="left") 

train_full = train_full.merge(transactions, on=["date", "store_nbr"], how="left")

# part 3 - building the features

train_full["year"] = train_full["date"].dt.year # extracts year from date
train_full["month"] = train_full["date"].dt.month # extracts month from date
train_full["day"] = train_full["date"].dt.day # extracts day from date
train_full["dayofweek"] = train_full["date"].dt.dayofweek  # monday=0, sunday=6
train_full["weekofyear"] = train_full["date"].dt.isocalendar().week.astype(int) # extracts week of year from date
train_full["is_weekend"] = (train_full["dayofweek"] >= 5).astype(int) # creates binary feature for weekend, which means dayofweek is 5 or 6

# sort by store/family/date before creating lags
train_full = train_full.sort_values(["store_nbr", "family", "date"]) # sorts the dataframe by store number, family, and date

# lag features
train_full["lag_1"] = train_full.groupby(["store_nbr", "family"])["sales"].shift(1) # 1 day lag
train_full["lag_7"] = train_full.groupby(["store_nbr", "family"])["sales"].shift(7) # 1 week lag
train_full["lag_14"] = train_full.groupby(["store_nbr", "family"])["sales"].shift(14) # 2 week lag

# rolling means
train_full["rolling_mean_7"] = ( # 7 day rolling mean
    train_full
    .groupby(["store_nbr", "family"])["sales"] # group by store and family and select sales column
    .shift(1)
    .rolling(window=7) 
    .mean()
)

train_full["rolling_mean_30"] = ( # 30 day rolling mean
    train_full
    .groupby(["store_nbr", "family"])["sales"]
    .shift(1)
    .rolling(window=30)
    .mean()
)

# holidays "transferred" flag: fill NaN, cast to int
train_full["transferred"] = train_full["transferred"].fillna(False) # fill NaN values with false
train_full["transferred"] = train_full["transferred"].astype(int) # convert boolean to int (false=0, true=1)

# label-encode all object columns
cat_cols = train_full.select_dtypes(include=["object"]).columns.tolist() # get list of categorical columns
le = LabelEncoder()

for col in cat_cols: # iterate through categorical columns
    train_full[col] = le.fit_transform(train_full[col].astype(str)) 
    # label encode each categorical column, which is essntially converting strings to integers

# part 4 - prepare data for modeling

# drop rows that have NaNs in any column (mostly from newly created lag/rolling features)
train_model = train_full.dropna().copy() 
# .copy() to avoid SettingWithCopyWarning which is when pandas is unsure if we're modifying a copy or the original dataframe

# feature list: everything except id, date, and target "sales"
feature_cols = [c for c in train_model.columns if c not in ["id", "date", "sales"]] 
# this works by list comprehension - iterates through all columns and includes them if not in the excluded list

X = train_model[feature_cols] # feature matrix
y = train_model["sales"] # target variable
dates = train_model["date"]  # for time-based split, which is when we split based on date rather than random sampling

# part 5 - time-based train/validation split

# 80% earliest dates -> train, latest 20% -> validation
cutoff_date = dates.quantile(0.8)
train_mask = dates <= cutoff_date
val_mask = dates > cutoff_date

X_train = X.loc[train_mask] # selects rows where train_mask is true
y_train = y.loc[train_mask] 
X_val = X.loc[val_mask] # selects rows where val_mask is true
y_val = y.loc[val_mask]

print(f"\nCutoff date for train/val split: {cutoff_date}") 
print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}") # prints number of samples in train and validation sets

# part 6 - train LightGBM model

# a strong baseline config (which Alex can tune later)
model = LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    num_leaves=63,
    max_depth=-1,
    random_state=67,
    n_jobs=-1,
)

# train model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)], # validation set for early stopping
    eval_metric="rmse",
)

# predictions on validation
val_pred = model.predict(X_val) 
val_pred = np.maximum(val_pred, 0)  # clamp negatives to zero (b/c sales can't be negative)

# RMSE (for reference)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"\nValidation RMSE: {rmse:.4f}")

# RMSLE 
rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))
print(f"Validation RMSLE: {rmsle:.4f}") # right now is 0.5795

# part 7 - save model and validation predictions

# save model for re-use
joblib.dump(model, "lightgbm_model.joblib") # this works by serializing the model object to a file

# attach predictions back onto the validation rows and save to CSV
val_rows = train_model.loc[val_mask].copy() # get original validation rows
val_rows["pred_sales"] = val_pred # add predictions as new column

# keep only a subset of columns that are useful
cols_to_save = ["date", "store_nbr", "family", "sales", "pred_sales"]
val_rows[cols_to_save].to_csv("val_predictions_lightgbm.csv", index=False) # save to CSV without index

print("\nSaved model to lightgbm_model.joblib")
print("Saved validation predictions to val_predictions_lightgbm.csv")