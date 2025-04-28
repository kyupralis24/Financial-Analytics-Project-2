# Financial Analytics Project 2: Option Pricing and Derivative Strategy Design

This project implements a Bull Call Spread options trading strategy analysis for Reliance Industries using real-time market data from NSE India.

## Features

- Real-time market data fetching from NSE CSV files
- Black-Scholes option pricing model implementation
- Monte Carlo simulation for option pricing
- Put-Call parity validation
- Profit/Loss analysis and break-even calculations
- Visual payoff diagram generation

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - scipy
  - pandas
  - matplotlib

## Data Source

The project uses real options data from NSE India (nseindia.com) in CSV format:
- Call Options Data: `OPTSTK_RELIANCE_CE_28-Jan-2025_TO_28-Apr-2025.csv`
- Put Options Data: `OPTSTK_RELIANCE_PE_28-Jan-2025_TO_28-Apr-2025.csv`

## Strategy Overview

The Bull Call Spread strategy involves:
- Buying one call option at the ATM (At-The-Money) strike price
- Selling one call option at the OTM (Out-of-The-Money) strike price
- Suitable for moderately bullish markets
- Limited risk and limited reward profile

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kyupralis24/Financial-Analytics-Project-2.git
cd Financial-Analytics-Project-2
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place NSE CSV files in the project directory:
   - `OPTSTK_RELIANCE_CE_28-Jan-2025_TO_28-Apr-2025.csv`
   - `OPTSTK_RELIANCE_PE_28-Jan-2025_TO_28-Apr-2025.csv`

## Usage

Run the script:
```bash
python proj2_1.py
```

The script will:
1. Read and process NSE CSV data
2. Calculate option prices using Black-Scholes model
3. Perform Monte Carlo simulation
4. Generate profit/loss analysis
5. Create a payoff diagram (saved as 'bull_call_spread_payoff.png')

## Output

The script provides detailed analysis including:
- Current market data
- Strategy rationale
- Option pricing analysis
- Profit/loss scenarios
- Break-even points
- Visual payoff diagram

## Analysis Components

1. **Market Data Analysis**:
   - Spot price extraction
   - Strike price selection
   - Premium calculation
   - Volatility estimation

2. **Pricing Models**:
   - Black-Scholes model implementation
   - Monte Carlo simulation
   - Put-Call parity validation

3. **Strategy Analysis**:
   - Break-even calculation
   - Maximum profit/loss
   - Risk-reward ratio
   - Payoff diagram generation

## Note

This project uses real market data from NSE India. The CSV files contain sensitive market data and should be handled appropriately. For demonstration purposes, the script uses historical volatility of 19.29% and a risk-free rate of 7.15% (91-day T-bill rate). 