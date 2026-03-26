"""
tools.py — All callable tools available to the TradingAgent.

Each tool is a plain Python function decorated with @tool_definition.
The agent decides which tools to call based on the user query or task.

Tools available:
  - get_trade_history      : Query closed trades from PostgreSQL
  - get_open_positions     : Current open trades
  - get_market_context     : Live regime from /market_context endpoint
  - get_strategy_stats     : Win rate / P&L breakdown per strategy
  - get_news               : Latest NSE announcements + Google RSS news
  - get_fii_dii_data       : Latest FII/DII institutional flow data
  - get_nifty_snapshot     : Nifty 50 current price + basic technicals
  - analyse_losing_trades  : Deep-dive on losing trades with pattern analysis

DISCLAIMER: For educational use only.
"""

import json
import logging
from datetime import date, timedelta
from typing import Any, Optional

import httpx
import pandas as pd
from sqlalchemy import text

from config.settings import get_settings
from trader_bot.risk_manager import get_engine

logger = logging.getLogger(__name__)
settings = get_settings()

# Registry — agent discovers tools from this dict
TOOL_REGISTRY: dict[str, callable] = {}


def register(fn):
    """Decorator to register a function as an agent tool."""
    TOOL_REGISTRY[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------------------
# Tool 1: Trade history
# ---------------------------------------------------------------------------

@register
def get_trade_history(days: int = 7, symbol: Optional[str] = None) -> dict:
    """
    Fetch closed trades from the database.

    Args:
        days:   How many days back to look (default 7).
        symbol: Filter by symbol (optional).
    """
    from_date = date.today() - timedelta(days=days)
    sql = "SELECT * FROM trades WHERE status='CLOSED' AND trade_date >= :from_date"
    params: dict = {"from_date": from_date}
    if symbol:
        sql += " AND symbol = :symbol"
        params["symbol"] = symbol.upper()
    sql += " ORDER BY created_at DESC LIMIT 200"

    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        if df.empty:
            return {"trades": [], "summary": "No closed trades in the period."}

        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
        summary = {
            "total_trades": len(df),
            "total_pnl": round(float(df["pnl"].sum()), 2),
            "win_rate_pct": round(len(df[df["pnl"] > 0]) / len(df) * 100, 1),
            "best_trade": round(float(df["pnl"].max()), 2),
            "worst_trade": round(float(df["pnl"].min()), 2),
        }
        trades = df[["trade_date", "symbol", "strategy", "side", "quantity",
                      "entry_price", "exit_price", "pnl"]].to_dict("records")
        return {"trades": trades, "summary": summary}
    except Exception as exc:
        logger.error("get_trade_history error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 2: Open positions
# ---------------------------------------------------------------------------

@register
def get_open_positions() -> dict:
    """Return all currently open (unfilled / running) trades from the database."""
    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM trades WHERE status='OPEN' ORDER BY created_at DESC"),
                conn,
            )
        return {"open_positions": df.to_dict("records") if not df.empty else []}
    except Exception as exc:
        logger.error("get_open_positions error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 3: Market context (calls local API)
# ---------------------------------------------------------------------------

@register
def get_market_context() -> dict:
    """
    Fetch the current market regime analysis from the TraderBot API.
    Returns regime, ADX, volatility, allowed strategies, and position size multiplier.
    """
    try:
        resp = httpx.get("http://localhost:8000/market_context", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("get_market_context error: %s", exc)
        return {"error": str(exc), "note": "TraderBot API may not be running."}


# ---------------------------------------------------------------------------
# Tool 4: Strategy performance stats
# ---------------------------------------------------------------------------

@register
def get_strategy_stats(days: int = 30) -> dict:
    """
    Break down win rate and average P&L per strategy over the last N days.

    Args:
        days: Look-back window (default 30).
    """
    from_date = date.today() - timedelta(days=days)
    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT strategy, pnl FROM trades "
                    "WHERE status='CLOSED' AND trade_date >= :from_date"
                ),
                conn,
                params={"from_date": from_date},
            )
        if df.empty:
            return {"stats": {}, "note": "No data in the period."}

        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
        stats = {}
        for strategy, grp in df.groupby("strategy"):
            wins = grp[grp["pnl"] > 0]
            stats[strategy] = {
                "total_trades": len(grp),
                "win_rate_pct": round(len(wins) / len(grp) * 100, 1),
                "avg_pnl": round(float(grp["pnl"].mean()), 2),
                "total_pnl": round(float(grp["pnl"].sum()), 2),
            }
        return {"stats": stats, "period_days": days}
    except Exception as exc:
        logger.error("get_strategy_stats error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 5: News fetcher (NSE + Google RSS)
# ---------------------------------------------------------------------------

@register
def get_news(symbol: Optional[str] = None) -> dict:
    """
    Fetch latest market news.
    If symbol is given, searches for that stock specifically.
    Otherwise returns general NSE/Indian market news.

    Args:
        symbol: Stock symbol e.g. 'RELIANCE', 'INFY' (optional).
    """
    query = f"{symbol} NSE India stock" if symbol else "NSE Nifty India stock market"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        resp = httpx.get(rss_url, timeout=10, follow_redirects=True)
        resp.raise_for_status()

        # Parse RSS manually (avoid feedparser dependency)
        import re
        items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
        headlines = []
        for item in items[:8]:
            title = re.search(r"<title>(.*?)</title>", item)
            pub_date = re.search(r"<pubDate>(.*?)</pubDate>", item)
            if title:
                headlines.append({
                    "title": title.group(1).strip(),
                    "published": pub_date.group(1).strip() if pub_date else "unknown",
                })
        return {"symbol": symbol or "market", "headlines": headlines}
    except Exception as exc:
        logger.error("get_news error: %s", exc)
        return {"error": str(exc), "headlines": []}


# ---------------------------------------------------------------------------
# Tool 6: FII/DII institutional flow
# ---------------------------------------------------------------------------

@register
def get_fii_dii_data() -> dict:
    """
    Fetch latest FII (Foreign Institutional Investors) and DII (Domestic)
    net buy/sell data from NSE India.
    Returns last 5 trading days of flow data.
    """
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com",
    }
    try:
        # NSE requires a session cookie — get it first
        session = httpx.Client(headers=headers, follow_redirects=True, timeout=15)
        session.get("https://www.nseindia.com")  # sets cookies
        resp = session.get(url)
        resp.raise_for_status()
        data = resp.json()

        # Extract last 5 days
        records = data[:5] if isinstance(data, list) else []
        result = []
        for r in records:
            result.append({
                "date": r.get("date", ""),
                "fii_net": r.get("fiiNet", 0),
                "dii_net": r.get("diiNet", 0),
                "market_bias": "BULLISH" if float(r.get("fiiNet", 0)) > 0 else "BEARISH",
            })
        return {"fii_dii": result}
    except Exception as exc:
        logger.warning("FII/DII fetch failed (NSE may block scraping): %s", exc)
        return {
            "error": str(exc),
            "note": "NSE API may require manual session. Check nseindia.com for latest data.",
        }


# ---------------------------------------------------------------------------
# Tool 7: Nifty snapshot (via TraderBot API quote)
# ---------------------------------------------------------------------------

@register
def get_nifty_snapshot() -> dict:
    """
    Return a quick snapshot of Nifty 50:
    current price, day change %, and 20-day EMA bias.
    Fetches from the local Kite connection.
    """
    try:
        from trader_bot.auth import get_kite_client
        kite = get_kite_client()
        quote = kite.quote("NSE:NIFTY 50")
        data = quote.get("NSE:NIFTY 50", {})
        last = data.get("last_price", 0)
        prev_close = data.get("ohlc", {}).get("close", last)
        change_pct = round((last - prev_close) / prev_close * 100, 2) if prev_close else 0
        return {
            "symbol": "NIFTY 50",
            "last_price": last,
            "change_pct": change_pct,
            "day_high": data.get("ohlc", {}).get("high", 0),
            "day_low": data.get("ohlc", {}).get("low", 0),
            "volume": data.get("volume", 0),
            "bias": "BULLISH" if change_pct > 0.3 else "BEARISH" if change_pct < -0.3 else "NEUTRAL",
        }
    except Exception as exc:
        logger.error("get_nifty_snapshot error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 8: Losing trade analyser
# ---------------------------------------------------------------------------

@register
def analyse_losing_trades(days: int = 14) -> dict:
    """
    Deep-dive on losing trades: find patterns in time-of-day,
    strategy, day-of-week, and stop-loss tightness.

    Args:
        days: Look-back window (default 14).
    """
    from_date = date.today() - timedelta(days=days)
    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT * FROM trades WHERE status='CLOSED' "
                    "AND pnl < 0 AND trade_date >= :from_date"
                ),
                conn,
                params={"from_date": from_date},
            )
        if df.empty:
            return {"analysis": "No losing trades in the period. Great job!"}

        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["day_of_week"] = df["trade_date"].dt.day_name()
        df["sl_distance_pct"] = (
            (df["entry_price"] - df["stop_loss"]).abs() / df["entry_price"] * 100
        ).round(2)

        patterns = {
            "total_losing_trades": len(df),
            "total_loss": round(float(df["pnl"].sum()), 2),
            "avg_loss_per_trade": round(float(df["pnl"].mean()), 2),
            "worst_strategy": df.groupby("strategy")["pnl"].sum().idxmin(),
            "worst_day_of_week": df["day_of_week"].value_counts().index[0],
            "avg_sl_distance_pct": round(float(df["sl_distance_pct"].mean()), 2),
            "tight_sl_trades": int((df["sl_distance_pct"] < 0.5).sum()),
            "by_strategy": df.groupby("strategy")["pnl"].agg(
                ["count", "sum", "mean"]
            ).round(2).to_dict(),
        }
        return {"patterns": patterns, "period_days": days}
    except Exception as exc:
        logger.error("analyse_losing_trades error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool dispatcher (used by agent)
# ---------------------------------------------------------------------------

def call_tool(name: str, arguments: dict) -> Any:
    """Dispatch a tool call by name with given arguments."""
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}


def get_tool_schemas() -> list[dict]:
    """Return OpenAI-compatible tool schema for all registered tools."""
    import inspect

    schemas = []
    for name, fn in TOOL_REGISTRY.items():
        sig = inspect.signature(fn)
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            has_default = param.default is not inspect.Parameter.empty
            annotation = param.annotation
            if annotation == int:
                ptype = "integer"
            elif annotation in (float,):
                ptype = "number"
            elif annotation in (bool,):
                ptype = "boolean"
            else:
                ptype = "string"

            properties[param_name] = {"type": ptype}
            # Mark Optional params with nullable
            if "Optional" in str(annotation):
                properties[param_name]["nullable"] = True
            if not has_default:
                required.append(param_name)

        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return schemas
