import pandas as pd
import numpy as np
import wrds

def load_and_prepare_data(csv_path):
    """
    Load WRDS Compustat Data and prepare it for analysis.
    
    Args:
        csv_path (str): Path to the WRDS NYSE TAQ dataset CSV file
    
    Returns:
        pd.DataFrame: Prepared IPO universe dataframe
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Convert date strings to datetime objects
    for col in ['ipodate', 'rdq', 'datadate']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Filter for valid IPOs (rows with an IPO Date)
    # Including Inactive companies to avoid survivorship bias
    ipo_universe = df[df['ipodate'].notnull()].copy()
    
    return ipo_universe

def get_peers(ipo_universe, target_ticker, n_peers=10):
    """
    Finds the top historical peers for a target IPO based on:
    - Same Sector (gsector)
    - Closest Market Cap (mkvaltq)
    - Strictly historical (Peer IPO Date < Target IPO Date)
    - Reporting Lag (Peer RDQ < Target IPO Date - 45 days)
    
    Args:
        ipo_universe (pd.DataFrame): DataFrame containing all IPO data
        target_ticker (str): Ticker symbol of the target IPO
        n_peers (int): Number of peers to return
    
    Returns:
        list: List of peer ticker symbols
    """
    target = ipo_universe[ipo_universe['tic'] == target_ticker].iloc[0]
    target_date = target['ipodate']
    target_sector = target['gsector']
    target_val = target['mkvaltq']

    # Potential peers: same sector, went public before target
    potential_peers = ipo_universe[
        (ipo_universe['gsector'] == target_sector) & 
        (ipo_universe['ipodate'] < target_date) &
        (ipo_universe['tic'] != target_ticker)
    ].copy()

    # Apply 45-day reporting lag: Peer must have reported fundamentals before target's IPO
    potential_peers = potential_peers[potential_peers['rdq'] < (target_date - pd.Timedelta(days=45))]

    if potential_peers.empty:
        return []

    # Calculate 'Valuation Distance'
    potential_peers['dist'] = (potential_peers['mkvaltq'] - target_val).abs()
    
    # Return top N closest peers by Market Cap
    return potential_peers.sort_values('dist')['tic'].head(n_peers).tolist()

def get_target_list(ipo_universe, sector_code=45, start_date='2024-01-01'):
    """
    Generate target list for prediction based on sector and date criteria.
    
    Args:
        ipo_universe (pd.DataFrame): DataFrame containing all IPO data
        sector_code (int): GICS sector code (default: 45 for Software)
        start_date (str): Start date for target selection
    
    Returns:
        numpy.ndarray: Array of target ticker symbols
    """
    targets = ipo_universe[
        (ipo_universe['gsector'] == sector_code) & 
        (ipo_universe['ipodate'] >= start_date)
    ]['tic'].unique()
    
    return targets

def build_peer_mapping(ipo_universe, targets, n_peers=10):
    """
    Build peer mapping for all target IPOs.
    
    Args:
        ipo_universe (pd.DataFrame): DataFrame containing all IPO data
        targets (numpy.ndarray): Array of target ticker symbols
        n_peers (int): Number of peers to find for each target
    
    Returns:
        dict: Mapping of target ticker to list of peer tickers
    """
    peer_map = {t: get_peers(ipo_universe, t, n_peers) for t in targets}
    return peer_map

def load_realized_volatility_from_wrds(target_ticker, peer_tickers, start_date, end_date, username=None):
    """
    Load realized volatility data from WRDS Cloud for target and peer stocks.
    
    Args:
        target_ticker (str): Target stock ticker
        peer_tickers (list): List of peer stock tickers
        start_date (str): Start date for data retrieval (YYYY-MM-DD)
        end_date (str): End date for data retrieval (YYYY-MM-DD)
        username (str): WRDS username (if None, will prompt for input)
    
    Returns:
        pd.DataFrame: DataFrame with 'Actual' and 'Peer_Prior' columns
    """
    try:
        # Connect to WRDS with fresh authentication
        db = wrds.Connection(wrds_username=username)
    except Exception as e:
        print(f"Failed to connect to WRDS: {e}")
        return None
    
    all_tickers = [target_ticker] + peer_tickers
    # Clean tickers - remove special characters like '.' and numbers
    cleaned_tickers = [ticker.split('.')[0] for ticker in all_tickers]
    rv_data = {}
    
    for ticker in cleaned_tickers:
        try:
            # First, get the permno for the ticker
            ticker_query = f"""
            SELECT DISTINCT permno, ticker
            FROM crsp_a_stock.stocknames
            WHERE ticker = '{ticker}'
            AND namedt <= '{end_date}'
            AND nameenddt >= '{start_date}'
            """
            
            ticker_data = db.raw_sql(ticker_query)
            if ticker_data.empty:
                print(f"No permno found for ticker {ticker}")
                continue
                
            permno = ticker_data['permno'].iloc[0]
            
            # Query CRSP daily data for realized volatility calculation
            query = f"""
            SELECT date, ret
            FROM crsp_a_stock.dsf 
            WHERE permno = {permno}
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
            """
            
            data = db.raw_sql(query)
            if not data.empty:
                # Calculate realized volatility as absolute daily returns
                data = data.dropna()
                if len(data) > 0:
                    rv_data[ticker] = data.set_index('date')['ret'].abs()  # Use absolute returns as RV proxy
                else:
                    print(f"No valid returns data for {ticker}")
            else:
                print(f"No data found for {ticker}")
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue
    
    db.close()
    
    # Create the final DataFrame
    if target_ticker in rv_data and len(rv_data) > 1:
        # Create target series
        target_series = rv_data[target_ticker]
        
        # Calculate peer average (excluding target)
        peer_series_list = [rv_data[ticker] for ticker in peer_tickers if ticker in rv_data]
        if peer_series_list:
            peer_avg = pd.concat(peer_series_list, axis=1).mean(axis=1)
        else:
            print("Warning: No peer data available")
            return None
        
        # Align and create final DataFrame
        final_df = pd.DataFrame({
            'Actual': target_series,
            'Peer_Prior': peer_avg.shift(1)  # Use lagged peer average
        }).dropna()
        
        return final_df
    else:
        print("Insufficient data to create forecasting dataset")
        return None

def create_sample_realized_volatility_data(start_date, end_date):
    """
    Create sample realized volatility data for testing when WRDS is not available.
    
    Args:
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: DataFrame with 'Actual' and 'Peer_Prior' columns
    """
    np.random.seed(42)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create realistic-looking realized volatility data
    # Use exponential distribution to simulate RV (always positive, with occasional spikes)
    target_data = np.random.exponential(0.02, len(dates))
    peer_data = np.random.exponential(0.018, len(dates))
    
    # Add some autocorrelation and volatility clustering
    for i in range(2, len(target_data)):
        target_data[i] = 0.7 * target_data[i-1] + 0.3 * target_data[i]
        peer_data[i] = 0.7 * peer_data[i-1] + 0.3 * peer_data[i]
    
    final_df = pd.DataFrame({
        'Actual': target_data,
        'Peer_Prior': np.roll(peer_data, 1)  # Lagged peer data
    }, index=dates)
    
    # Remove first row due to lag and any remaining NaN values
    final_df = final_df.iloc[1:].dropna()
    
    print(f"✅ Created sample RV data with {len(final_df)} observations")
    return final_df

if __name__ == "__main__":
    # Example usage
    csv_path = '/Users/krishsapru/Downloads/WRDS_NYSE_TAQ_dataset.csv'
    
    # Load and prepare data
    ipo_universe = load_and_prepare_data(csv_path)
    
    # Generate target list
    targets = get_target_list(ipo_universe)
    print(f"Found {len(targets)} target IPOs for forecasting.")
    
    # Build peer mapping (testing with first target)
    peer_map = build_peer_mapping(ipo_universe, targets[:1])
    
    # Display results
    for target, peers in peer_map.items():
        print(f"Target: {target} | Peers: {', '.join(peers)}")
