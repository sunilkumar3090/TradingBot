"""
celery_app.py — Celery worker and Beat schedule for EvaluatorBot + TradingAgent.

Tasks:
  - evaluate_trades_task:  runs every 30 min during market hours
  - daily_summary_task:    runs at 15:45 IST (10:15 UTC)
  - morning_briefing_task: runs at 9:00 AM IST (3:30 UTC)
  - post_trade_explain_task: triggered manually after a trade closes
"""

from celery import Celery
from celery.schedules import crontab

from config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "evaluator_bot",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    beat_schedule={
        # Morning briefing at 9:00 AM IST (3:30 UTC) Mon-Fri
        "morning-briefing": {
            "task": "evaluator_bot.celery_app.morning_briefing_task",
            "schedule": crontab(hour=3, minute=30, day_of_week="1-5"),
        },
        # Run every 30 minutes Mon-Fri during market hours (9:15–15:30 IST)
        "evaluate-every-30min": {
            "task": "evaluator_bot.celery_app.evaluate_trades_task",
            "schedule": crontab(minute="*/30", hour="9-15", day_of_week="1-5"),
        },
        # Daily summary at 15:45 IST (10:15 UTC)
        "daily-summary": {
            "task": "evaluator_bot.celery_app.daily_summary_task",
            "schedule": crontab(hour=10, minute=15, day_of_week="1-5"),
        },
    },
)


@celery_app.task(name="evaluator_bot.celery_app.morning_briefing_task", bind=True, max_retries=2)
def morning_briefing_task(self):
    """AI-generated morning briefing sent to Telegram at 9:00 AM IST."""
    try:
        from trading_agent.briefing import run_morning_briefing
        run_morning_briefing()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=120)


@celery_app.task(name="evaluator_bot.celery_app.evaluate_trades_task", bind=True, max_retries=3)
def evaluate_trades_task(self):
    """Periodic intraday evaluation task."""
    try:
        from evaluator_bot.evaluator import run_evaluation
        return run_evaluation()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="evaluator_bot.celery_app.daily_summary_task", bind=True, max_retries=2)
def daily_summary_task(self):
    """End-of-day AI advisory and Telegram notification."""
    try:
        from evaluator_bot.ai_advisor import run_daily_advisor
        run_daily_advisor()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=120)


@celery_app.task(name="evaluator_bot.celery_app.post_trade_explain_task", bind=True, max_retries=2)
def post_trade_explain_task(self, trade: dict):
    """
    Explain a completed trade using the TradingAgent.
    Call this after closing a trade by passing the trade dict.

    Usage:
        post_trade_explain_task.delay(trade_dict)
    """
    try:
        from trading_agent.briefing import run_post_trade_explanation
        run_post_trade_explanation(trade)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
