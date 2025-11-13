import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.tsa.stattools import adfuller 
from sklearn.metrics import mean_squared_error 
import numpy as np 
# Load dataset 
df = pd.read_csv("C:\\Users\\prava\\Downloads\\retail_sales_dataset.csv", 
parse_dates=['Date']) 
df 
# Convert to datetime and sort 
df['Date'] = pd.to_datetime(df['Date']) 
df = df.sort_values('Date') 
df 
# Calculate daily total demand per product category 
df['Total Amount'] = df['Quantity'] * df['Price per Unit'] 
daily_demand = df.groupby(['Date', 'Product Category'])['Quantity'].sum().reset_index() 
daily_demand 
# Let's choose one category for forecasting (e.g., 'Electronics') 
category = 'Electronics' 
category_demand = daily_demand[daily_demand['Product Category'] == category] 
category_demand 
# Set date as index and fill missing dates 
category_demand = category_demand.set_index('Date').resample('D').sum().fillna(0) 
category_demand 
#. Visualize Time Series 
category_demand['Quantity'].plot(figsize=(12, 5), title=f'{category} Demand Over Time') 
plt.ylabel('Quantity Sold') 
plt.xlabel('Date') 
plt.show() 
#Check Stationarity (ADF Test) 
result = adfuller(category_demand['Quantity']) 
print(f'ADF Statistic: {result[0]}') 
print(f'p-value: {result[1]}') 
if result[1] > 0.05: 
category_demand['Quantity_diff'] = category_demand['Quantity'].diff().dropna() 
#Train-Test Split 
train = category_demand.iloc[:-30]  # All except last 30 days 
test = category_demand.iloc[-30:]   # Last 30 days 
train 
#Train-Test Split 
train = category_demand.iloc[:-30]  # All except last 30 days 
test = category_demand.iloc[-30:]   # Last 30 days 
test 
# Fit SARIMA Model 
model = SARIMAX(train['Quantity'], order=(1,1,1), seasonal_order=(1,1,1,7))  # Weekly 
seasonality 
model_fit = model.fit(disp=False)  
# Forecast 
forecast = model_fit.forecast(steps=30) 
forecast 
#Plot Forecast vs Actual 
plt.figure(figsize=(12,5)) 
plt.plot(train.index, train['Quantity'], label='Train') 
plt.plot(test.index, test['Quantity'], label='Actual') 
plt.plot(test.index, forecast, label='Forecast', linestyle='--') 
plt.title(f'Demand Forecast for {category}') 
plt.xlabel('Date') 
plt.ylabel('Quantity') 
plt.legend() 
plt.show() 
#Evaluate Performance 
mse = mean_squared_error(test['Quantity'], forecast) 
print(f'Mean Squared Error: {mse:.2f}')