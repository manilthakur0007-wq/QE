# Quant Engine — Research & Paper Trading

## Resume & GitHub-ready summary
--------------------------------

This repository provides a modular quantitative equity research and paper-trading
system implemented in Python. It is designed for simulation and paper trading
only and must be configured with legitimate market data providers/APIs.
- Project: `quant_engine` — a modular quantitative equity research and paper-trading
  system. Designed for simulated capital management, backtesting, and paper trading via Alpaca.
- Highlights: modular data ingestion (yfinance/Alpaca), async streaming adapter, multi-timeframe
  feature engineering (momentum, ATR, volatility), hybrid multi-factor strategy, volatility-based
  position sizing, event-like backtester, and a safe Alpaca paper trading connector.
- Languages & libs: Python 3.11+, Pandas, NumPy, Scikit-learn, Alpaca SDK.

How this fits on your resume:
- Implemented a modular research pipeline supporting historical ingestion, live streaming,
  feature engineering, risk-management, and execution (paper-only).
- Built event-driven backtester with realistic fills, commission and slippage modeling.
- Enabled end-to-end reproducibility with config-driven parameters, tests and CI.

Uploading to GitHub
- Add your `.env` with Alpaca keys (or leave empty to run simulated mode).
- Initialize git, commit, and push. Example:

```bash
git init
git add .
git commit -m "Initial quant_engine project"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```


Key features
- Modular architecture: data, features, strategy, risk, execution, backtest
- Historical ingestion + async streaming adapter
- Feature engineering: multi-timeframe momentum, ATR, volatility, liquidity
- Hybrid multi-factor strategy with optional ML classifier
- Risk management: volatility sizing, stop loss, drawdown cutoff
- Event-like backtest engine with realistic fills, slippage and commission
- Alpaca paper trading connector (only in paper mode)

Requirements
- Python 3.11+
- See `requirements.txt` for dependencies. Install with:

```bash
python -m pip install -r requirements.txt
Notes
```

Setup
1. Copy `.env.example` to `.env` and add your Alpaca API keys.
2. Edit `config.yaml` to adjust strategy and risk parameters.

Running
- Backtest (default sample run using `main.py`):

```bash
python main.py --mode backtest
```

- Paper trading (requires Alpaca paper API keys):

```bash
python main.py --mode paper
```

Notes
- The system defaults to $10,000 simulated capital and operates in paper/simulated
  mode by default. Do not use this code in production without review.
