import wrds
import pandas as pd
import numpy as np

def test_wrds_connection():
    """Test WRDS connection and create sample data if needed"""
    try:
        print("Testing WRDS connection...")
        db = wrds.Connection(wrds_username='krishsapru')
        print("✅ WRDS connection successful!")
        
        # Test a simple query
        query = "SELECT COUNT(*) as count FROM crsp_a_stock.dsf LIMIT 1"
        result = db.raw_sql(query)
        print(f"✅ Query successful: {result['count'].iloc[0]} records available")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ WRDS connection failed: {e}")
        return False

def create_sample_data():
    """Create sample realized volatility data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    # Create sample data for VHAI and some peers
    target_data = np.random.exponential(0.02, len(dates))  # Simulate RV with exponential distribution
    peer_data = np.random.exponential(0.018, len(dates))
    
    final_df = pd.DataFrame({
        'Actual': target_data,
        'Peer_Prior': np.roll(peer_data, 1)  # Lagged peer data
    }, index=dates)
    
    # Remove first row due to lag
    final_df = final_df.iloc[1:]
    
    print(f"✅ Created sample data with {len(final_df)} observations")
    return final_df

if __name__ == "__main__":
    if test_wrds_connection():
        print("WRDS is available - you can use real data")
    else:
        print("WRDS not available - using sample data")
        sample_df = create_sample_data()
        print(sample_df.head())
