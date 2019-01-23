
#recession predictor
#predict % likelihood of recession over next 8 quarters
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
recession = 'USREC' # binary.  NBER based recession indicators for the united states from the period following the peak to the trough.
stockMarket = 'SPASTT01USM661N' #Total Share Prices for All Shares for the United States 
durableGoods = 'CAPUTLG3361T3SQ' #motor vehicles and parts
manufacturingOrders = 'ODMNTO02USQ470S' #value of total orders for manufacturing in US
inventoriesDelta = 'A015RX1Q020SBEA'#change in private inventories Nonfarm
inventoriesDeltaTotal = 'A014RY2Q224SBEA' #change in private inventories total
retailSales = 'SLRTTO01USQ661S' #volume of total US retail sales
housingPermits = 'PERMIT' #new private housing building permits
housePrice = 'MSPUS' #median housing sale price US
leadingIndex = 'USSLIND' #leading index for US
employment = 'USTRADE' #retail trade employment

variables = [recession, stockMarket, durableGoods, manufacturingOrders,
            inventoriesDelta, inventoriesDeltaTotal,
            retailSales, housingPermits, housePrice, 
            leadingIndex, employment]

names = ['recession', 'stockMarket', 'durableGoods', 'manufacturingOrders',
            'inventoriesDelta', 'inventoriesDeltaTotal',
            'retailSales', 'housingPermits', 'housePrice', 
            'leadingIndex', 'employment']




#Using fredapi to download pandas series, converting this to a dictionary of dataframes

d = {}
for item in variables:
    print(f'looking up data for {item}')
    data = fred.get_series(item)
    name = names[variables.index(item)]
    d[name] = pd.DataFrame(data, columns = [name])
    
d.keys() #returns the names of the dataframes within the dictionary d.  we will organize them next





#organize data (maybe recession is boolean, rather than GDP delta), turn all data quarterly





#random forest model





#train model





#optimize on test data





#predict % likelihood of recession over next 8 quarters



#visualize forecast
