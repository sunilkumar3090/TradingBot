"""
ai_advisor.py — AI-powered post-market analysis report.

Supports two backends:
  Option A: Kite MCP Server (https://mcp.kite.trade/mcp) — zero-install AI access
  Option B: OpenAI GPT (requires OPENAI_API_KEY in .env)

DISCLAIMER: AI output is NOT financial advice.
            For educational use only. All financial risk is assumed by the user.
"""

import json
import logging
from datetime import date
from typing import Optional

from config.settings import get_settings
from evaluator_bot.evaluator import fetch_trades, run_evaluation

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Telegram notification helper
# ---------------------------------------------------------------------------

def send_telegram_message(message: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.info("Telegram not configured — skipping notification.")
        return False
    try:
        import httpx
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        resp = httpx.post(
            url,
            json={"chat_id": settings.telegram_chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Telegram notification sent.")
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Option A: Kite MCP Server
# ---------------------------------------------------------------------------

MCP_SERVER_CONFIG = {
    "mcpServers": {
        "kite-trader": {
            "command": "npx",
            "args": ["mcp-remote", "https://mcp.kite.trade/mcp"],
        }
    }
}


def get_mcp_config() -> dict:
    """Return MCP server config for AI assistant integration."""
    return MCP_SERVER_CONFIG


# ---------------------------------------------------------------------------
# AI report generation — respects EVALUATOR_LLM_PROVIDER setting
# Supported: github (default, free) | deepseek | openai | ollama
# ---------------------------------------------------------------------------

def _build_evaluator_client():
    """Build an OpenAI-compatible client for the evaluator provider."""
    from openai import OpenAI
    provider = getattr(settings, "evaluator_llm_provider", "github").lower()

    if provider == "github":
        if not settings.github_token:
            raise RuntimeError(
                "GITHUB_TOKEN not set. Add your GitHub PAT to .env — "
                "free 150 req/day via GitHub Models."
            )
        logger.info("Evaluator LLM: GitHub Models (%s)", settings.github_model)
        return (
            OpenAI(api_key=settings.github_token, base_url=settings.github_models_base_url),
            settings.github_model,
        )

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set.")
        logger.info("Evaluator LLM: DeepSeek (%s)", settings.deepseek_model)
        return (
            OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url),
            settings.deepseek_model,
        )

    if provider == "ollama":
        logger.info("Evaluator LLM: Ollama (%s)", settings.ollama_model)
        return (
            OpenAI(api_key="ollama", base_url=settings.ollama_base_url),
            settings.ollama_model,
        )

    # Default fallback: OpenAI
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    logger.info("Evaluator LLM: OpenAI (%s)", settings.openai_model)
    return (
        OpenAI(api_key=settings.openai_api_key),
        settings.openai_model,
    )


def generate_ai_report(metrics: dict, sample_trades: list) -> Optional[str]:
    """
    Generate post-market analysis using the configured evaluator LLM provider.
    Defaults to GitHub Models (free, uses your GitHub PAT).
    Returns the AI-generated report string, or None on failure.
    """
    try:
        client, model = _build_evaluator_client()
        prompt = _build_prompt(metrics, sample_trades)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional quantitative trading analyst. "
                        "Provide concise, actionable post-market analysis. "
                        "Do NOT provide personalised investment advice. "
                        "Highlight strategy weaknesses and risk management observations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.4,
        )
        provider = getattr(settings, "evaluator_llm_provider", "github")
        report = response.choices[0].message.content
        logger.info("AI report generated via %s.", provider)
        return report
    except RuntimeError as exc:
        logger.warning("Evaluator LLM not configured: %s — skipping AI report.", exc)
        return None
    except Exception as exc:
        logger.error("AI report generation failed: %s", exc)
        return None


# Keep old name as alias for backward compatibility
generate_openai_report = generate_ai_report


def _build_prompt(metrics: dict, sample_trades: list) -> str:
    trades_summary = json.dumps(sample_trades[:10], default=str, indent=2)
    return f"""
Post-Market Analysis Report — {date.today()}

Performance Metrics:
- Total Trades: {metrics.get('total_trades', 'N/A')}
- Win Rate: {metrics.get('win_rate', 'N/A')}%
- Total P&L: ₹{metrics.get('total_pnl', 'N/A')}
- Profit Factor: {metrics.get('profit_factor', 'N/A')}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}
- Max Drawdown: ₹{metrics.get('max_drawdown', 'N/A')}

Sample Trades (last 10):
{trades_summary}

Please provide:
1. A brief analysis of today's trading performance.
2. Key reasons why trades succeeded or failed (e.g., stop loss sizing, market conditions).
3. Specific, actionable recommendations to improve the strategy.
4. Any risk management concerns to address.
"""


# ---------------------------------------------------------------------------
# Main advisor entry point
# ---------------------------------------------------------------------------

def run_daily_advisor(target_date: Optional[date] = None) -> None:
    """
    Run evaluation + AI analysis and send Telegram notification.
    Called by Celery Beat at 3:45 PM IST.
    """
    target_date = target_date or date.today()
    metrics = run_evaluation(target_date)

    if not metrics:
        msg = f"*TraderBot Daily Summary — {target_date}*\n\nNo closed trades today."
        send_telegram_message(msg)
        return

    # Build base summary message
    summary = (
        f"*TraderBot Daily Summary — {target_date}*\n\n"
        f"Trades: {metrics['total_trades']} | "
        f"Win Rate: {metrics['win_rate']}%\n"
        f"P&L: ₹{metrics['total_pnl']} | "
        f"Profit Factor: {metrics['profit_factor']}\n"
        f"Sharpe: {metrics['sharpe_ratio']} | "
        f"Max DD: ₹{metrics['max_drawdown']}"
    )

    # Optionally append AI narrative
    df = fetch_trades(target_date)
    trades_list = df.to_dict("records") if not df.empty else []
    ai_report = generate_ai_report(metrics, trades_list)
    if ai_report:
        summary += f"\n\n*AI Analysis:*\n{ai_report}"

    send_telegram_message(summary)
    logger.info("Daily advisor complete for %s.", target_date)
