"""
ChatEngine: Conversational query engine with database retrieval.
Answers user questions using historical indicator and prediction data.
"""

import logging
from typing import Dict, Optional, Any, List

from src.ai.engine import AIEngine
from src.ai.cache import AICache
from src.ai.prompts import chat_system_prompt, chat_context_prompt
from src.ai.schemas import ChatResponse
from src.database import CryptoDatabase
from src.config import settings

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Conversational AI that answers technical analysis questions.
    Retrieves recent data from database for context.
    """

    def __init__(self, engine: Optional[AIEngine] = None, cache: Optional[AICache] = None):
        self.engine = engine or AIEngine()
        self.cache = cache or AICache()

    def ask(
        self,
        query: str,
        coin_id: str = "bitcoin",
        context_hours: int = 24,
        db: Optional[CryptoDatabase] = None,
    ) -> ChatResponse:
        """
        Answer a user query with database-backed context.

        Args:
            query: User question
            coin_id: Coin to query about
            context_hours: How many hours of history to include
            db: Database instance (created if None)

        Returns:
            ChatResponse Pydantic model
        """
        # Get database instance
        db = db or CryptoDatabase(settings.DB_PATH)

        # Retrieve context from database
        recent_indicators = self._get_recent_indicators(db, coin_id, context_hours)
        recent_predictions = self._get_recent_predictions(db, coin_id, 10)

        # Build prompts
        system = chat_system_prompt()
        prompt = chat_context_prompt(
            coin_id=coin_id,
            recent_indicators=recent_indicators,
            recent_predictions=recent_predictions,
            query=query,
        )

        # Check cache (chat is less cacheable, but simple repeated queries benefit)
        cache_key = f"{coin_id}:{query}"
        cached = self.cache.get(cache_key, "chat", coin_id)
        if cached:
            return ChatResponse(**cached)

        # Generate via LLM
        result = self.engine.generate(
            prompt=prompt,
            system=system,
            schema_class=ChatResponse,
        )

        # Cache result
        self.cache.set(cache_key, "chat", result, coin_id)

        return ChatResponse(**result)

    def _get_recent_indicators(
        self,
        db: CryptoDatabase,
        coin_id: str,
        hours: int,
    ) -> List[Dict[str, Any]]:
        """Fetch recent indicator rows for context."""
        try:
            df = db.get_indicator_window(coin_id=coin_id, lookback=50)
            if df.empty:
                return []
            # Convert to list of dicts, most recent first
            records = df.head(10).to_dict('records')
            return records
        except Exception as e:
            logger.warning(f"Failed to fetch indicators for chat context: {e}")
            return []

    def _get_recent_predictions(
        self,
        db: CryptoDatabase,
        coin_id: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Fetch recent prediction history for context."""
        try:
            df = db.get_predictions(coin_id=coin_id, limit=limit)
            if df.empty:
                return []
            records = df.to_dict('records')
            return records
        except Exception as e:
            logger.warning(f"Failed to fetch predictions for chat context: {e}")
            return []