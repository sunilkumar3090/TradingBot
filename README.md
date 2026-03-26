# TradingBot — AI-Powered Intraday Trading System

> **DISCLAIMER:** This project is for **educational purposes only**. Trading involves significant financial risk. The developer assumes **no liability** for any financial losses. Always paper-trade before going live.

---

## Architecture

```
trading-bot/
├── trader_bot/
│   ├── auth.py          # OAuth flow, daily token refresh
│   ├── data_stream.py   # KiteTicker WebSocket → Redis
│   ├── strategies.py    # MA Crossover, Momentum (RSI), VCP
│   ├── execution.py     # Order placement, bracket orders, rate limiting
│   ├── risk_manager.py  # Position sizing, drawdown gate, DB logging
│   └── main.py          # FastAPI app + trading loop
├── evaluator_bot/
│   ├── evaluator.py     # P&L, Sharpe, Win Rate, Max Drawdown
│   ├── ai_advisor.py    # OpenAI / Kite MCP post-market report
│   └── celery_app.py    # Celery Beat scheduler
├── config/
│   └── settings.py      # Pydantic settings from .env
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
| `GET` | `/status` | Health check |
| `POST` | `/start` | Start trading loop |
| `POST` | `/stop` | Emergency stop (cancels all orders) |
| `GET` | `/positions` | View current positions |
| `POST` | `/manual_order` | Place a manual order |

Interactive docs at: `http://localhost:8000/docs`

---

## Strategies

| Strategy | Logic | Confidence |
|---|---|---|
| `MA_Crossover` | Fast EMA crosses above/below slow EMA | 0.6 |
| `Momentum_RSI` | RSI crosses oversold on a volume spike | 0.7 |
| `VCP` | Volatility Contraction Pattern breakout | 0.8 |

---

## Risk Controls

- **Position sizing**: Fixed-fraction (1% risk per trade by default)
- **Daily drawdown gate**: Halts trading if cumulative loss exceeds 3% of capital
- **Rate limiting**: Max 100 Kite API calls/minute (token-bucket algorithm)
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
- Every 30 minutes (Mon–Fri, 9:15–15:30 IST): performance metrics
- 15:45 IST daily: AI-generated post-market report → Telegram

---

## Security Notes

- API keys loaded from `.env` only — never hardcoded
- Token cache stored with `chmod 600` permissions
- Docker container runs as non-root user
- `.gitignore` excludes `.env` and token cache file
