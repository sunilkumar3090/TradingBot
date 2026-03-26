"""
agent.py — Core TradingAgent with tool-calling loop.

Supports three LLM providers via LLM_PROVIDER in .env:
  - deepseek : DeepSeek-V3 / R1 (recommended — cheap, powerful, OpenAI-compatible)
  - openai   : GPT-4o / GPT-4o-mini
  - ollama   : Local LLM via Ollama (free, offline, OpenAI-compatible)

All providers use the OpenAI client with different base_url + api_key.
It runs a ReAct-style loop:
  Thought → Tool call → Observe result → Thought → ... → Final answer

DISCLAIMER: AI output is NOT financial advice. For educational use only.
"""

import json
import logging
from typing import Optional

from openai import OpenAI

from config.settings import get_settings
from trading_agent.tools import call_tool, get_tool_schemas

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _build_client() -> tuple[OpenAI, str]:
    """
    Build an OpenAI-compatible client for the configured LLM provider.
    Returns (client, model_name).
    """
    provider = settings.llm_provider.lower()

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. "
                "Get a free key at https://platform.deepseek.com and add it to .env"
            )
        logger.info("LLM provider: DeepSeek (%s)", settings.deepseek_model)
        return (
            OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url),
            settings.deepseek_model,
        )

    if provider == "ollama":
        logger.info("LLM provider: Ollama (%s) — local, free", settings.ollama_model)
        return (
            OpenAI(api_key="ollama", base_url=settings.ollama_base_url),
            settings.ollama_model,
        )

    if provider == "github":
        if not settings.github_token:
            raise RuntimeError(
                "GITHUB_TOKEN not set. Add your GitHub PAT to .env."
            )
        logger.info("LLM provider: GitHub Models (%s)", settings.github_model)
        return (
            OpenAI(api_key=settings.github_token, base_url=settings.github_models_base_url),
            settings.github_model,
        )

    # Default: OpenAI
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to .env or switch LLM_PROVIDER to deepseek/ollama/github."
        )
    logger.info("LLM provider: OpenAI (%s)", settings.openai_model)
    return (
        OpenAI(api_key=settings.openai_api_key),
        settings.openai_model,
    )

# System prompt — defines the agent's persona and constraints
SYSTEM_PROMPT = """You are TradingAgent, an expert AI trading analyst for the Indian stock market.

You have access to tools that query real trading data, market context, news, and institutional flow.

Your responsibilities:
1. Analyse trade performance and explain WHY trades succeeded or failed
2. Identify patterns in losing trades (wrong day, tight stop, bad regime, news events)
3. Provide actionable recommendations to improve strategy performance
4. Give morning market briefings with a clear directional bias
5. Answer trader questions using real data — never guess

Rules you MUST follow:
- Always use tools to fetch data before answering — never hallucinate numbers
- Cite specific data points (e.g. "Win rate dropped to 38% on Fridays vs 65% on Tuesdays")
- Keep answers concise and actionable — no fluff
- Always remind the user this is NOT financial advice
- If tools return errors, acknowledge it and work with available data

Indian market context:
- NSE trading hours: 9:15 AM – 3:30 PM IST
- Key indices: Nifty 50, Bank Nifty
- FII/DII flow is a leading indicator for next-day bias
- Budget/expiry weeks have different volatility patterns
"""


class TradingAgent:
    """
    Stateless tool-calling agent. Call .run(query) to get a response.
    Conversation history is maintained per-session externally.

    LLM provider is selected via LLM_PROVIDER in .env:
      LLM_PROVIDER=deepseek  → DeepSeek-V3 (default, recommended)
      LLM_PROVIDER=openai    → GPT-4o-mini
      LLM_PROVIDER=ollama    → local Ollama model
    """

    def __init__(self, max_iterations: int = 6):
        self.client, self.model = _build_client()
        self.max_iterations = max_iterations
        self.tools = get_tool_schemas()

    def run(
        self,
        query: str,
        history: Optional[list[dict]] = None,
    ) -> tuple[str, list[dict]]:
        """
        Run the agent on a user query.

        Args:
            query:   The user's question or task.
            history: Prior conversation messages (for multi-turn chat).

        Returns:
            (answer_text, updated_history)
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1200,
            )

            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            # No tool calls — agent has a final answer
            if not message.tool_calls:
                answer = message.content or ""
                # Return updated history (excluding system prompt)
                return answer, messages[1:]

            # Execute all tool calls in this turn
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info("Agent calling tool: %s(%s)", fn_name, fn_args)
                result = call_tool(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                })

        # Fallback if max iterations hit
        logger.warning("Agent hit max iterations (%d) without a final answer.", self.max_iterations)
        return "I was unable to complete the analysis in the allowed steps. Please try a more specific question.", messages[1:]


# ---------------------------------------------------------------------------
# Convenience singleton — lazy-initialised
# ---------------------------------------------------------------------------

_agent_instance: Optional[TradingAgent] = None


def get_agent() -> TradingAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TradingAgent()
    return _agent_instance
