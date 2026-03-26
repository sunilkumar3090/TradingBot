"""
chat.py — Interactive Q&A FastAPI router for the TradingAgent.

Mounts at /agent prefix in the main TraderBot API.

Endpoints:
  POST /agent/chat    — single-turn or multi-turn chat with the agent
  POST /agent/ask     — quick single question, no history
  GET  /agent/briefing — trigger morning briefing on demand

DISCLAIMER: AI responses are NOT financial advice. For educational use only.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/agent", tags=["TradingAgent"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[ChatMessage]] = None   # pass prior turns for multi-turn


class ChatResponse(BaseModel):
    answer: str
    history: list[ChatMessage]    # updated history — pass back on next request


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Multi-turn chat with the TradingAgent.
    Pass history from prior responses to maintain context.

    Example questions:
    - "Why did I lose money last Thursday?"
    - "Which strategy has the best win rate this month?"
    - "Is today a good day to trade aggressively?"
    - "Analyse my losing trades from the past 2 weeks"
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not configured. Add it to .env to use the TradingAgent.",
        )
    try:
        from trading_agent.agent import get_agent
        agent = get_agent()

        history = [m.model_dump() for m in req.history] if req.history else []
        answer, updated_history = agent.run(req.message, history=history)

        return ChatResponse(
            answer=answer,
            history=[ChatMessage(**m) for m in updated_history
                     if m.get("role") in ("user", "assistant") and m.get("content")],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Agent error — check logs.")


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Single-turn quick question. No conversation history maintained.

    Best for:
    - "What is the current market regime?"
    - "Give me a summary of today's trades"
    - "What's the FII flow today?"
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not configured. Add it to .env to use the TradingAgent.",
        )
    try:
        from trading_agent.agent import get_agent
        agent = get_agent()
        answer, _ = agent.run(req.question)
        return AskResponse(answer=answer)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Ask endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Agent error — check logs.")


@router.get("/briefing")
def get_briefing():
    """
    Trigger a morning briefing on demand.
    Returns the briefing text and sends it to Telegram.
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured.")
    try:
        from trading_agent.briefing import run_morning_briefing
        briefing = run_morning_briefing()
        if briefing:
            return {"briefing": briefing}
        raise HTTPException(status_code=500, detail="Briefing generation failed.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Briefing endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
