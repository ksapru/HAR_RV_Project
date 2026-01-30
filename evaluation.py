import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_forecast(results_df):
    """
    Evaluate forecast performance using RMSE and skill score.
    
    Args:
        results_df (pd.DataFrame): DataFrame with 'Actual' and 'Forecast' columns
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Calculate Forecast Error
    rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Forecast']))
    
    # Calculate Naive Error (Baseline)
    naive_forecast = results_df['Actual'].shift(1).fillna(method='bfill')
    naive_rmse = np.sqrt(mean_squared_error(results_df['Actual'], naive_forecast))
    
    improvement = (naive_rmse - rmse) / naive_rmse
    
    metrics = {
        'model_rmse': rmse,
        'naive_rmse': naive_rmse,
        'skill_score': improvement
    }
    
    print(f"Model RMSE: {rmse:.6f}")
    print(f"Naive RMSE: {naive_rmse:.6f}")
    print(f"Skill Score (Improvement): {improvement:.2%}")
    
    return metrics
