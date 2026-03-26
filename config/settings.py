"""
Configuration settings loaded from environment variables.
NEVER hardcode API keys or secrets in this file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # --- Kite Connect Credentials ---
    kite_api_key: str = Field(..., env="KITE_API_KEY")
    kite_api_secret: str = Field(..., env="KITE_API_SECRET")
    kite_redirect_uri: str = Field(
        default="http://localhost:3000/api/auth/callback",
        env="KITE_REDIRECT_URI",
    )

    # --- Database ---
    database_url: str = Field(..., env="DATABASE_URL")

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # --- Celery ---
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/1", env="CELERY_RESULT_BACKEND")

    # --- Notifications ---
    telegram_bot_token: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", env="TELEGRAM_CHAT_ID")

    # --- OpenAI (optional, for AI advisor) ---
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")

    # --- Risk Parameters ---
    max_risk_per_trade_pct: float = Field(default=1.0, env="MAX_RISK_PER_TRADE_PCT")   # 1% of capital
    max_daily_drawdown_pct: float = Field(default=3.0, env="MAX_DAILY_DRAWDOWN_PCT")   # 3% of capital

    # --- Trading Parameters ---
    trading_exchange: str = Field(default="NSE", env="TRADING_EXCHANGE")
    order_product_type: str = Field(default="MIS", env="ORDER_PRODUCT_TYPE")  # MIS = intraday
    kite_api_rate_limit: int = Field(default=100, env="KITE_API_RATE_LIMIT")  # orders per minute

    # --- Environment ---
    environment: str = Field(default="development", env="ENVIRONMENT")  # development | production

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
