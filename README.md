# Financial Analytics Project 2: Option Pricing and Derivative Strategy Design

This project implements a Long Straddle options trading strategy analysis for Reliance Industries using real-time market data from Alpha Vantage API.

## Features

- Real-time market data fetching for Reliance Industries
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
  - requests

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Financial-Analytics-Project-2.git
cd Financial-Analytics-Project-2
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the script:
```bash
python proj2_1.py
```

The script will:
1. Fetch real-time market data for Reliance Industries
2. Calculate option prices using Black-Scholes model
3. Perform Monte Carlo simulation
4. Generate profit/loss analysis
5. Create a payoff diagram (saved as 'long_straddle_payoff.png')

## Output

The script provides detailed analysis including:
- Current market data
- Strategy rationale
- Option pricing analysis
- Profit/loss scenarios
- Break-even points
- Visual payoff diagram

## Note

This project uses Alpha Vantage API for market data. The API key is included in the script for demonstration purposes. For production use, consider using environment variables or a configuration file to store the API key securely. 