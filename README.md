# HAR-RV IPO Volatility Forecasting Project

**Log-Linear Realized GARCH Model for IPO Volatility Prediction Using Peer Group Analysis**

## Overview

This project implements a sophisticated HAR-RV (Heterogeneous Autoregressive Realized Volatility) forecasting model specifically designed for predicting IPO (Initial Public Offering) volatility using peer group analysis and WRDS Cloud data integration.

## Key Features

- **WRDS Cloud Integration**: Direct connection to WRDS database for real-time CRSP data
- **Peer Mapping Algorithm**: Intelligent peer selection based on sector and market cap
- **Rolling Window Forecasting**: Walk-forward validation with HAR-RV model
- **Performance Evaluation**: Comprehensive metrics vs naive benchmark
- **Sample Data Fallback**: Robust testing when WRDS is unavailable

## Project Structure

```
HAR_RV_Project/
├── main.py                 # Main pipeline orchestration
├── data_loading.py         # WRDS integration and data processing
├── forecasting.py          # HAR-RV model implementation
├── evaluation.py           # Performance metrics and benchmarks
├── test_wrds.py           # WRDS connectivity testing
├── requirements.txt       # Python dependencies
├── .gitignore            # Git exclusions
└── README.md             # This file
```

## Methodology

### 1. Data Pipeline
- Loads IPO universe from WRDS Compustat data
- Maps target IPOs to peer groups based on sector and size
- Fetches realized volatility from CRSP daily stock data
- Calculates peer average volatility as fundamental predictor

### 2. HAR-RV Model
```
RV_t+1 = β₀ + β₁·RV_t + β₂·Peer_Prior_t + ε_t+1
```

- **Target RV**: Absolute daily returns of IPO stock
- **Peer Prior**: Lagged average volatility of peer group
- **Rolling Window**: 6-period training window for real-time simulation

### 3. Evaluation Framework
- **RMSE**: Root Mean Squared Error
- **Naive Benchmark**: Random walk forecast
- **Skill Score**: Percentage improvement over naive

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ksapru/HAR_RV_Project.git
cd HAR_RV_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up WRDS credentials:
- Ensure you have WRDS access
- The script will prompt for username and password

## Usage

### Basic Execution
```bash
python3 main.py
```

### Configuration
Edit `main.py` to customize:
- Sector code (default: 45 for Software)
- Date range (default: 2024)
- Window size (default: 6 periods)
- WRDS username

### Sample Data Mode
If WRDS is unavailable, the pipeline automatically falls back to realistic sample data for testing.

## Example Output

```
Loading and preparing data...
Found 47 target IPOs for forecasting.
Building peer mapping...
Target: VHAI | Peers: ADCT.1, ADI, CNLG, MIKR, MANA., PLAB, XETA, DELL, VSTI, CSCO

Loading realized volatility data from WRDS Cloud...
Successfully loaded 138 observations for VHAI

Running HAR-RV forecasting...
Evaluating forecast performance...
Model RMSE: 0.251
Naive RMSE: 0.104
Skill Score (Improvement): -141.67%
```

## Performance Results

Current implementation shows:
- **Data Coverage**: 138 daily observations for target IPO
- **Peer Success**: 5/10 peers with valid CRSP data
- **Model Performance**: Baseline HAR-RV implementation
- **Improvement Opportunities**: Feature engineering, parameter tuning

## Technical Architecture

### Data Sources
- **WRDS Compustat**: IPO universe and fundamentals
- **CRSP Daily Stock**: Price data for volatility calculation
- **Stock Names Database**: Ticker to permno mapping

### Model Components
- **Peer Selection**: Sector + market cap filtering
- **Volatility Proxy**: Absolute daily returns
- **Rolling Forecast**: Walk-forward validation
- **Benchmark**: Random walk with drift

## Dependencies

- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `statsmodels>=0.13.0` - Econometric modeling
- `scikit-learn>=1.1.0` - Machine learning utilities
- `wrds>=3.4.0` - WRDS Cloud connection

## Future Enhancements

1. **Feature Engineering**
   - Volume-based volatility measures
   - Intraday high-low range
   - Volatility clustering indicators

2. **Model Improvements**
   - GARCH family extensions
   - Machine learning hybrids
   - Multi-horizon forecasting

3. **Data Expansion**
   - Cross-asset peer groups
   - Macroeconomic factors
   - Sentiment indicators

4. **Production Features**
   - Real-time data pipelines
   - Automated model selection
   - Performance monitoring

## Research Applications

This framework is suitable for:
- **Quantitative Finance**: Volatility trading strategies
- **Academic Research**: IPO market microstructure
- **Risk Management**: Post-IPO volatility forecasting
- **Portfolio Construction**: Peer-based factor models

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **WRDS** for providing comprehensive financial data
- **CRSP** for high-quality stock price data
- **Compustat** for fundamental company information

## Contact

Krish Sapru - [GitHub](https://github.com/ksapru)

---

**Note**: This implementation is for research and educational purposes. Production usage requires additional validation, risk management, and compliance considerations.
