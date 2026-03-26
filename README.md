# TradingBot — AI-Powered Intraday Trading System

> **DISCLAIMER:** This project is for **educational purposes only**. Trading involves significant financial risk. The developer assumes **no liability** for any financial losses. Always paper-trade before going live.

---

## Architecture

```
trading-bot/
├── trader_bot/
│   ├── auth.py            # OAuth flow, daily token refresh
│   ├── data_stream.py     # KiteTicker WebSocket → Redis tick cache
│   ├── strategies.py      # 5 strategies: ORB, VWAP, Momentum, VCP, MA Crossover
│   ├── market_context.py  # Market regime detection (ADX + volatility + Nifty bias)
│   ├── execution.py       # Order placement, bracket orders, rate limiting
│   ├── risk_manager.py    # Position sizing, drawdown gate, DB logging
│   └── main.py            # FastAPI app + regime-aware trading loop
├── evaluator_bot/
│   ├── evaluator.py       # P&L, Sharpe, Win Rate, Max Drawdown
│   ├── ai_advisor.py      # OpenAI / Kite MCP post-market report + Telegram
│   └── celery_app.py      # Celery Beat scheduler
├── config/
│   └── settings.py        # Pydantic settings from .env
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── Dockerfile
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Zerodha Kite Connect account | Paid subscription for live WebSocket data |
| API Key + Secret | From [kite.trade/connect](https://kite.trade/connect) |
| Python 3.11+ | Or use Docker |
| PostgreSQL 15 | For trade storage |
| Redis 7 | For tick cache + Celery broker |

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/sunilkumar3090/TradingBot.git
cd TradingBot
cp .env.example .env
# Edit .env with your Kite API credentials
```

### 2. Run with Docker (recommended)

```bash
docker-compose up -d
```

All services (TraderBot API, Celery worker, Beat scheduler, PostgreSQL, Redis) start automatically.

### 3. Run locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL and Redis separately, then:
uvicorn trader_bot.main:app --reload --port 8000
```

---

## Authentication (Daily Step)

Kite requires a fresh `access_token` every trading day:

```bash
# 1. Get the login URL
python -c "from trader_bot.auth import get_login_url; print(get_login_url())"

# 2. Open the URL, log in, copy the request_token from the redirect URL

# 3. Exchange it for an access token
python -c "
from trader_bot.auth import generate_session
generate_session('PASTE_REQUEST_TOKEN_HERE')
"
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Health check + bot state |
| `GET` | `/market_context` | Live market regime analysis |
| `POST` | `/start` | Start the trading loop |
| `POST` | `/stop` | Emergency stop — cancels all open orders |
| `GET` | `/positions` | View current intraday positions |
| `POST` | `/manual_order` | Place a manual order |

Interactive docs: `http://localhost:8000/docs`

---

## Strategies

Strategies run in priority order every 5 minutes. Only strategies compatible with the current market regime are executed.

| Priority | Strategy | Signal Logic | Confidence |
|---|---|---|---|
| 1 | `ORB_Bullish` / `ORB_Bearish` | Break above/below first-15-min range on 1.5x volume | 0.75–0.78 |
| 2 | `VWAP_Bounce` | Price dips to VWAP lower band and recovers with volume | 0.72 |
| 2 | `VWAP_Retest_Support` | Price retests VWAP as support in a trend | 0.75 |
| 2 | `VWAP_Rejection` | Price rejected at VWAP upper band (short) | 0.70 |
| 3 | `Momentum_RSI` | RSI crosses oversold threshold on a volume spike | 0.70 |
| 4 | `VCP` | Volatility Contraction Pattern breakout | 0.80 |
| 5 | `MA_Crossover` | Fast EMA crosses above/below slow EMA | 0.60 |

---

## Market Regime Detection

Before every trade cycle the bot classifies the market using Nifty 50 daily data:

| Regime | Condition | Position Size | Allowed Strategies |
|---|---|---|---|
| `TRENDING_BULL` | ADX ≥ 25, price > EMA20 | 100% | ORB, VWAP Retest, Momentum, VCP, MA |
| `TRENDING_BEAR` | ADX ≥ 25, price < EMA20 | 80% | ORB short, VWAP Rejection, MA |
| `MEAN_REVERTING` | ADX < 25 | 70% | VWAP Bounce, VWAP Retest, Momentum |
| `HIGH_VOLATILITY` | VIX > 20 or ATR ratio > 1.8 | 40% | VWAP Bounce only |
| `UNKNOWN` | Insufficient data | 0% | None — trading blocked |

Check live regime at: `GET /market_context`

---

## Risk Controls

- **Position sizing**: Fixed-fraction (1% risk per trade by default), scaled by regime multiplier
- **Daily drawdown gate**: Halts trading if cumulative loss exceeds 3% of capital
- **Rate limiting**: Max 100 Kite API calls/minute (token-bucket algorithm)
- **One trade per symbol per cycle**: Prevents over-trading a single instrument
- **Never commit secrets**: `.env` is in `.gitignore`

---

## Celery Tasks

```bash
# Start worker
celery -A evaluator_bot.celery_app worker --loglevel=info

# Start scheduler
celery -A evaluator_bot.celery_app beat --loglevel=info
```

Schedule:
- Every 30 minutes (Mon–Fri, 9:15–15:30 IST): compute and store performance metrics
- 15:45 IST daily: AI post-market report (OpenAI GPT) → Telegram notification

---

## Security Notes

- API keys loaded from `.env` only — never hardcoded
- Token cache stored with `chmod 600` permissions
- Docker container runs as non-root user
- `.gitignore` excludes `.env` and token cache file
