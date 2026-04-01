# Digital Twin Investment Advisory System

Thesis project: "Design and Implementation of an Intelligent Investment Advisory System
Inspired by the Digital Twin Concept"

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py

# 3. Open http://localhost:8501
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## File Structure

```
thesis-project/
├── app.py                        # Streamlit dashboard (entry point)
├── requirements.txt
├── data/
│   ├── portfolio_moderate.csv    # Sample: balanced portfolio
│   ├── portfolio_aggressive.csv  # Sample: growth portfolio
│   ├── portfolio_conservative.csv# Sample: income portfolio
│   └── scenarios_config.json     # Pre-defined scenario definitions
├── src/
│   ├── data_loader.py            # Fetch data from Yahoo Finance (yfinance)
│   ├── portfolio.py              # Digital Twin — Portfolio class + all metrics
│   ├── risk_monitor.py           # 6 risk checks → alerts
│   ├── scenario_engine.py        # What-if simulation engine
│   ├── recommendations.py        # Explainable rebalancing recommendations
│   ├── explainer.py              # Plain-English explanations (XAI layer)
│   └── utils.py                  # build_portfolio() + formatting helpers
└── tests/
    ├── test_portfolio.py
    ├── test_risk_monitor.py
    ├── test_scenario_engine.py
    ├── test_recommendations.py
    └── test_data_loader.py
```

## System Components

| Component | Module | Purpose |
|-----------|--------|---------|
| Digital Twin | `portfolio.py` | Virtual replica of portfolio with all metrics |
| Scenario Engine | `scenario_engine.py` | Simulate market conditions (crash, rates, etc.) |
| Risk Monitor | `risk_monitor.py` | Identify 6 categories of portfolio risk |
| Recommendations | `recommendations.py` | Explainable rebalancing suggestions |
| XAI Layer | `explainer.py` | Plain-English explanations for everything |
| Dashboard | `app.py` | Streamlit interactive UI |

## Portfolio CSV Format

```csv
ticker,quantity,entry_price
AAPL,100,150.00
MSFT,50,300.00
SPY,200,380.00
BND,500,80.00
```

## Data Source

All market data fetched from Yahoo Finance via `yfinance`.
Requires internet connection for live prices and historical data.
