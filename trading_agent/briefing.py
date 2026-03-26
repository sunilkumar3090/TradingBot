"""
briefing.py — Daily morning briefing and post-trade explainer.

Morning briefing (runs at 9:00 AM IST via Celery):
  - Fetches overnight FII/DII flow, Nifty snapshot, news
  - Agent synthesises a directional bias + top watchlist stocks
  - Sends to Telegram

Post-trade explainer (triggered after each trade closes):
  - Explains WHY the trade won or lost
  - Flags if the signal was valid in hindsight
  - Suggests stop-loss / entry adjustments

DISCLAIMER: For educational use only. Not financial advice.
"""

import logging
from datetime import date
from typing import Optional

from config.settings import get_settings
from evaluator_bot.ai_advisor import send_telegram_message
from trading_agent.agent import TradingAgent

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Morning Briefing
# ---------------------------------------------------------------------------

MORNING_BRIEFING_PROMPT = """
Generate a morning trading briefing for today ({today}).

Please:
1. Call get_nifty_snapshot to get current Nifty price and bias
2. Call get_fii_dii_data to check institutional flow direction
3. Call get_news to get top market headlines
4. Call get_market_context to get the current regime

Then provide:
- **Market Bias**: Bullish / Bearish / Neutral with reasoning
- **Key Risk**: One sentence — what could go wrong today
- **Strategy Recommendation**: Which of our strategies to favour (ORB / VWAP / Momentum / VCP)
- **Watchlist Tip**: Any sector or stock theme to focus on
- **Position Sizing**: Recommended size multiplier based on conditions

Keep it under 200 words. Be direct and actionable.
"""

POST_TRADE_PROMPT = """
Analyse this recently closed trade and explain what happened:

Trade details:
- Symbol: {symbol}
- Strategy: {strategy}
- Side: {side}
- Entry: ₹{entry_price}
- Exit: ₹{exit_price}
- Stop Loss: ₹{stop_loss}
- P&L: ₹{pnl}
- Date: {trade_date}

Please:
1. Call get_news(symbol="{symbol}") to check if any news affected the stock
2. Call get_market_context to see what the regime was
3. Call get_strategy_stats(days=30) to check if this strategy is performing well overall

Then explain:
- Was this a valid signal or a bad setup?
- What caused the trade to win/lose?
- One specific improvement for next time (tighter stop? wider target? avoid this regime?)

Be concise — 3-4 sentences max.
"""


def run_morning_briefing() -> Optional[str]:
    """
    Generate and send morning briefing via Telegram.
    Called by Celery Beat at 9:00 AM IST.
    """
    if not settings.openai_api_key:
        logger.info("OpenAI not configured — skipping morning briefing.")
        return None

    try:
        agent = TradingAgent(max_iterations=5)
        prompt = MORNING_BRIEFING_PROMPT.format(today=date.today())
        answer, _ = agent.run(prompt)

        message = f"*🌅 Morning Briefing — {date.today()}*\n\n{answer}"
        send_telegram_message(message)
        logger.info("Morning briefing sent.")
        return answer
    except Exception as exc:
        logger.error("Morning briefing failed: %s", exc)
        return None


def run_post_trade_explanation(trade: dict) -> Optional[str]:
    """
    Explain a completed trade using the agent.
    Pass a trade dict from the trades DB table.
    Sends result to Telegram and returns explanation text.
    """
    if not settings.openai_api_key:
        return None

    try:
        agent = TradingAgent(max_iterations=4)
        prompt = POST_TRADE_PROMPT.format(
            symbol=trade.get("symbol", "N/A"),
            strategy=trade.get("strategy", "N/A"),
            side=trade.get("side", "N/A"),
            entry_price=trade.get("entry_price", 0),
            exit_price=trade.get("exit_price", 0),
            stop_loss=trade.get("stop_loss", 0),
            pnl=trade.get("pnl", 0),
            trade_date=trade.get("trade_date", date.today()),
        )
        answer, _ = agent.run(prompt)

        pnl = float(trade.get("pnl", 0))
        emoji = "✅" if pnl >= 0 else "❌"
        symbol = trade.get("symbol", "")
        message = (
            f"*{emoji} Trade Closed — {symbol}*\n"
            f"P&L: ₹{pnl:+.2f}\n\n"
            f"*AI Analysis:*\n{answer}"
        )
        send_telegram_message(message)
        logger.info("Post-trade explanation sent for %s.", symbol)
        return answer
    except Exception as exc:
        logger.error("Post-trade explanation failed: %s", exc)
        return None
