# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 1.11.2025
### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load avocado dataset
# -----------------------------
data = pd.read_csv('/content/avocado.csv')

# Convert Date column to datetime format and sort
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.sort_values('Date')

# Set Date as index
data.set_index('Date', inplace=True)

# -----------------------------
# Function for ARIMA modeling
# -----------------------------
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data', color='blue')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data', color='green')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.grid()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# -----------------------------
# Run ARIMA model on Total Volume
# -----------------------------
arima_model(data, 'Total Volume', order=(5, 1, 0))
```
### OUTPUT:
<img width="719" height="466" alt="image" src="https://github.com/user-attachments/assets/1acb536c-2f32-49ec-ac0c-1e5c3df5ba14" />
### RESULT:
Thus the program run successfully based on the ARIMA model using python.
