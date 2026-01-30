from data_loading import load_and_prepare_data, get_target_list, build_peer_mapping, load_realized_volatility_from_wrds, create_sample_realized_volatility_data
from forecasting import rolling_forecast
from evaluation import evaluate_forecast

def main():
    """
    Main execution function for HAR RV model pipeline.
    """
    # Configuration
    csv_path = '/Users/krishsapru/Downloads/WRDS_NYSE_TAQ_dataset.csv'
    sector_code = 45  # Software sector
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    window_size = 6
    wrds_username = None  # Set your WRDS username here, or leave None to be prompted
    
    # 1. Load and prepare data
    print("Loading and preparing data...")
    ipo_universe = load_and_prepare_data(csv_path)
    
    # 2. Generate target list
    targets = get_target_list(ipo_universe, sector_code, start_date)
    print(f"Found {len(targets)} target IPOs for forecasting.")
    
    # 3. Build peer mapping (testing with first target for demonstration)
    print("Building peer mapping...")
    peer_map = build_peer_mapping(ipo_universe, targets[:1])  # Testing with first target
    
    # Display peer mapping
    for target, peers in peer_map.items():
        print(f"Target: {target} | Peers: {', '.join(peers)}")
    
    # 4. Load realized volatility data from WRDS Cloud
    print("\nLoading realized volatility data from WRDS Cloud...")
    target_ticker = list(peer_map.keys())[0]  # Use first target for demonstration
    peer_tickers = peer_map[target_ticker]
    
    final_df = load_realized_volatility_from_wrds(
        target_ticker=target_ticker,
        peer_tickers=peer_tickers,
        start_date=start_date,
        end_date=end_date,
        username=wrds_username
    )
    
    # Fallback to sample data if WRDS fails
    if final_df is None:
        print("WRDS data loading failed, using sample data...")
        final_df = create_sample_realized_volatility_data(start_date, end_date)
    
    if final_df is not None:
        print(f"Successfully loaded {len(final_df)} observations for {target_ticker}")
        
        # 5. Run forecasting
        print("\nRunning HAR-RV forecasting...")
        results = rolling_forecast(final_df['Actual'], final_df['Peer_Prior'], window_size=window_size)
        
        # 6. Evaluate forecasts
        print("\nEvaluating forecast performance...")
        metrics = evaluate_forecast(results)
        print(f"Forecast Metrics: {metrics}")
        
        return ipo_universe, targets, peer_map, results, metrics
    else:
        print("Failed to load realized volatility data")
        return ipo_universe, targets, peer_map, None, None

if __name__ == "__main__":
    ipo_universe, targets, peer_map, results, metrics = main()
