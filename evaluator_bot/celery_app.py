"""
celery_app.py — Celery worker and Beat schedule for EvaluatorBot.

Tasks:
  - evaluate_trades_task: runs every 30 min during market hours
  - daily_summary_task:   runs at 15:45 IST (10:15 UTC)
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
