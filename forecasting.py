import pandas as pd
import statsmodels.api as sm

def rolling_forecast(target_rv_series, peer_avg_rv_series, window_size=6):
    """
    Simulates a real-time forecasting environment.
    window_size=6 assumes 5-min intervals (30 min training lookback).
    
    Args:
        target_rv_series (pd.Series): Target realized volatility series
        peer_avg_rv_series (pd.Series): Peer average realized volatility series
        window_size (int): Size of the rolling training window
    
    Returns:
        pd.DataFrame: DataFrame with actual values and forecasts
    """
    # Align target data with the fundamental peer average
    data = pd.DataFrame({
        'Actual_RV': pd.to_numeric(target_rv_series, errors='coerce'),
        'Lag_RV': pd.to_numeric(target_rv_series.shift(1), errors='coerce'),
        'Peer_Prior': pd.to_numeric(peer_avg_rv_series, errors='coerce')
    }).dropna()
    
    # Ensure all data is numeric
    data = data.astype(float)
    
    print(f"Data shape: {data.shape}")
    print(f"Data types: {data.dtypes}")
    print(f"Sample data: {data.head()}")

    predictions = []
    observations = []

    # The Walk-Forward Loop
    for t in range(window_size, len(data)):
        train = data.iloc[t-window_size:t]
        test = data.iloc[t:t+1]

        # 1. Retrain model on the sliding window
        X_train = train[['Lag_RV', 'Peer_Prior']]
        X_train = sm.add_constant(X_train)
        y_train = train['Actual_RV']
        model = sm.OLS(y_train, X_train).fit()

        # 2. Forecast the next interval (t + 1)
        X_test = test[['Lag_RV', 'Peer_Prior']]
        X_test = sm.add_constant(X_test, has_constant='add')
        forecast = model.predict(X_test)

        predictions.append(forecast.values[0])
        observations.append(test['Actual_RV'].values[0])

    return pd.DataFrame({
        'Actual': observations, 
        'Forecast': predictions
    }, index=data.index[window_size:])
