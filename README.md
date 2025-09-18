# Store-Sales

Based on the Kaggle Competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)


## Description of the Data

#### `train.csv`

Time series of features 
- `store_nbr`: the store at which the products are sold
- `family`: the type of product sold
- `sales`: total sales for that product family at a given store at a given date. **target feature**. 
- `onpromotion`: the total number of items in product family that were being promoted at a given store at a given date.
#### `test.csv`

Has same features as `train.csv`, minus the target feature `sales`. The dates in the test data are for 15 days after the last date in the training data. 

#### `sample_submission.csv`

A sample dataset in the correct format for submission.

#### `stores.csv`

Store metadata, including 
- `store_nbr` : store number
- `city`
- `state`
- `type`
- `cluster` (grouping of similar stores)

#### `transactions.csv`

Contains the number of transactions at each store on each date
- `date`
- `store_nbr`
- `transactions`

#### `oil.csv`

Daily oil prices.

#### `holiday_events.csv`
