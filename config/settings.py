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

    # --- LLM Provider (openai | deepseek | ollama) ---
    llm_provider: str = Field(default="deepseek", env="LLM_PROVIDER")

    # --- OpenAI ---
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")

    # --- DeepSeek (recommended — cheap + powerful) ---
    deepseek_api_key: str = Field(default="", env="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", env="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", env="DEEPSEEK_BASE_URL")

    # --- Ollama (local, free, offline) ---
    ollama_model: str = Field(default="llama3.2", env="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434/v1", env="OLLAMA_BASE_URL")

    # --- GitHub Models (free, uses GitHub PAT — 150 req/day) ---
    github_token: str = Field(default="", env="GITHUB_TOKEN")
    github_model: str = Field(default="gpt-4o-mini", env="GITHUB_MODEL")
    github_models_base_url: str = Field(default="https://models.inference.ai.azure.com", env="GITHUB_MODELS_BASE_URL")

    # --- Evaluator LLM Provider (can differ from trading agent) ---
    evaluator_llm_provider: str = Field(default="github", env="EVALUATOR_LLM_PROVIDER")

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
