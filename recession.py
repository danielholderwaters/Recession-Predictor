#recession predictor
#predict % likelihood of recession
import requests
import json
import pandas as pd
from fredapi import Fred
#get data: quarterly gdp data, leading economic indicators
#all data can probably be found on FRED database.  to get an api key, register here: https://research.stlouisfed.org/docs/api/fred/
#Using the fredapi wrapper, the data is returned 
from config import fredKey
fred = Fred(api_key=fredKey)

#data sources
recession = 'USREC' # monthly. binary.  NBER based recession indicators for the united states from the period following the peak to the trough.
stockMarket = 'SPASTT01USM661N' # monthly. Total Share Prices for All Shares for the United States 
durableGoods = 'CAPUTLG3361T3SQ' #quarterly. motor vehicles and parts
                                 # monthly. CAPUTLG3361T3S   
manufacturingOrders = 'ODMNTO02USQ470S' #quarterly. value of total orders for manufacturing in US
                                #no monthly #
inventoriesDelta = 'A015RX1Q020SBEA'# quarterly. change in private inventories Nonfarm
inventoriesDeltaTotal = 'A014RY2Q224SBEA' #quarterly. change in private inventories total
retailSales = 'SLRTTO01USQ661S' #quarterly. volume of total US retail sales
                                #monthly  Sales: Retail trade: Total retail trade: Volume for the United States (USASLRTTO01GYSAM)
housingPermits = 'PERMIT' #monthly. new private housing building permits
housePrice = 'MSPUS' #quarterly. median housing sale price US
                        #monthly Median Sales Price for New Houses Sold in the United States (MSPNHSUS)
leadingIndex = 'USSLIND' #monthly. leading index for US
employment = 'USTRADE' #monthly. retail trade employment

variables = [recession, stockMarket, durableGoods, manufacturingOrders,
            inventoriesDelta, inventoriesDeltaTotal,
            retailSales, housingPermits, housePrice, 
            leadingIndex, employment]

names = ['recession', 'stockMarket', 'durableGoods', 'manufacturingOrders',
            'inventoriesDelta', 'inventoriesDeltaTotal',
            'retailSales', 'housingPermits', 'housePrice', 
            'leadingIndex', 'employment']

#Using fredapi to download pandas series, converting this to a dictionary of dataframes, turning data quarterly
d = []
for item in variables:
    print(f'looking up data for {item}')
    data = fred.get_series(item)
    name = names[variables.index(item)]
    #this line creates the dataframe 
    d.append(pd.DataFrame({'date': data.index, name : data.values}))
    

#organize data (maybe recession is boolean, rather than GDP delta) and averages the value for each quarter, with the date beginning on the first day of each quarter
dffull = pd.concat([a.set_index('date') for a in d], axis=1).resample('QS').mean()
df = pd.concat([a.set_index('date') for a in d], axis=1).resample('QS').mean().dropna()
#after removing all of the NA values, 146 quarters as of 1/29/19.  15 quarters are recessions ~10% of the time
#creating negatively lagged values of recession in order to forecast recessions in the next 4 years

df["rec1Q"] = df['recession'].shift(-1).fillna(0)
df["rec2Q"] = df['recession'].shift(-2).fillna(0)
df["rec3Q"] = df['recession'].shift(-3).fillna(0)
df["rec4Q"] = df['recession'].shift(-4).fillna(0)
df.tail()

#random forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

y=df.rec2Q #recession next quarter
features = ['stockMarket', 'durableGoods', 'manufacturingOrders',
            'inventoriesDelta', 'inventoriesDeltaTotal',
            'retailSales', 'housingPermits', 'housePrice', 
            'leadingIndex', 'employment', 'rec1Q']
X = df[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Specify the model
recessionModel2Q = RandomForestRegressor(random_state=1)

# Fit iowa_model with the training data.
recessionModel2Q.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = recessionModel2Q.predict(val_X)
#Compare predictive capability in terms of mean absolute error
rf_val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

#chart results
import matplotlib.pyplot as plt
plt.scatter(val_predictions, val_y, s=10)
plt.xlabel("predicted liklihood of recession")
plt.ylabel("Observed State (1 is Recession)")
print(plt.show())

