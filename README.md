# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Path Finder Logic for Colab
file_path = next((p for p in ['/content/aapl_master_enriched.csv', 'aapl_master_enriched.csv'] if os.path.exists(p)), None)

if file_path:
    # 2. Explore the Dataset
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    # Resample to Monthly average to handle seasonality (s=12)
    data = df['close'].resample('MS').mean().dropna()

    print("--- OUTPUT: DATA EXPLORATION ---")
    print(data.head())

    # 3. Stationarity Check
    print("\n--- STATIONARITY CHECK (ADF TEST) ---")
    result = adfuller(data)
    print(f'ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}')

    # 4. Determine Parameters (ACF/PACF Plots)
    # Using 1st order differencing to visualize potential p,q values
    data_diff = data.diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(data_diff, ax=axes[0], lags=24, title='ACF (Monthly Differenced)')
    plot_pacf(data_diff, ax=axes[1], lags=24, title='PACF (Monthly Differenced)')
    plt.show()

    # 5. Fit SARIMA Model
    # Splitting: Train on historical data, test on the last 12 months
    train = data.iloc[:-12]
    test = data.iloc[-12:]

    # Parameters: (p,d,q) x (P,D,Q,s)
    # Common starting point for seasonal data: order=(1,1,1), seasonal=(1,1,1,12)
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    print("\n--- MODEL SUMMARY ---")
    print(model.summary())

    # 6. Make Predictions
    forecast_obj = model.get_forecast(steps=12)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    # 7. Evaluate Performance
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"\n--- EVALUATION ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Plot final results
    plt.figure(figsize=(12, 6))
    plt.plot(train.tail(24), label='Training Data')
    plt.plot(test, label='Actual Price', color='black', marker='o')
    plt.plot(test.index, forecast, label='SARIMA Forecast', color='red', linestyle='--', marker='x')
    plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('Apple Stock Price: SARIMA Model Prediction (12-Month Forecast)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```
### OUTPUT:
<img width="780" height="402" alt="image" src="https://github.com/user-attachments/assets/6b0be078-edb2-4d21-a214-51341de0530f" />
<img width="445" height="308" alt="image" src="https://github.com/user-attachments/assets/956332f4-83f4-4901-96fc-9c5d86ca727a" />
<img width="618" height="384" alt="image" src="https://github.com/user-attachments/assets/d68602c8-6ead-4b86-995f-96fc8175e83d" />
### RESULT:
Thus the program run successfully based on the SARIMA model.
