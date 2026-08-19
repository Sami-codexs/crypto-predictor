"""
MarketReporter: Generates daily/periodic market briefings combining
prediction, indicators, and drift status into a cohesive narrative.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

from src.ai.engine import AIEngine
from src.ai.cache import AICache
from src.ai.prompts import market_briefing_prompt, SYSTEM_PROMPT
from src.ai.schemas import MarketBriefing

logger = logging.getLogger(__name__)


class MarketReporter:
    """
    Generates AI-powered market briefings.
    Caches briefings per hour to avoid redundant LLM calls.
    """

    def __init__(self, engine: Optional[AIEngine] = None, cache: Optional[AICache] = None):
        self.engine = engine or AIEngine()
        self.cache = cache or AICache()

    def generate(
        self,
        coin_id: str,
        prediction: Dict[str, Any],
        indicators: Dict[str, Any],
        drift_summary: Optional[Dict[str, Any]] = None,
    ) -> MarketBriefing:
        """Generate a market briefing for a coin."""
        
        # Handle None prediction (not enough data)
        if prediction is None:
            prediction = {
                'prediction': 'neutral',
                'confidence': 0.5,
                'probability': 0.5,
                'timestamp': datetime.now().isoformat(),
            }
        
        if indicators is None:
            indicators = {
                'coin': coin_id,
                'timestamp': datetime.now().isoformat(),
                'indicators': {}
            }

        # Build prompt
        prompt = market_briefing_prompt(
            coin_id=coin_id,
            prediction=prediction,
            indicators=indicators,
            drift_summary=drift_summary,
        )

        # Check cache
        cached = self.cache.get(prompt, "briefing", coin_id)
        if cached:
            return MarketBriefing(**cached)

        # Generate via LLM
        result = self.engine.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            schema_class=MarketBriefing,
        )

        # Handle both dict and MarketBriefing returns
        if isinstance(result, MarketBriefing):
            self.cache.set(prompt, "briefing", result.dict(), coin_id)
            return result
        
        # Cache and return
        self.cache.set(prompt, "briefing", result, coin_id)
        return MarketBriefing(**result)