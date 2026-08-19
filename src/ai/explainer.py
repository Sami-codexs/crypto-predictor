"""
PredictionExplainer: Generates natural language explanations for
LSTM predictions using technical indicators only.
"""

import logging
from typing import Dict, Optional, Any

from src.ai.engine import AIEngine
from src.ai.cache import AICache
from src.ai.prompts import prediction_explanation_prompt, SYSTEM_PROMPT
from src.ai.schemas import PredictionExplanation

logger = logging.getLogger(__name__)


class PredictionExplainer:
    """
    Explains why the model made a specific prediction.
    Uses cached responses to avoid redundant LLM calls.
    """

    def __init__(self, engine: Optional[AIEngine] = None, cache: Optional[AICache] = None):
        self.engine = engine or AIEngine()
        self.cache = cache or AICache()

    def explain(
        self,
        coin_id: str,
        prediction: Dict[str, Any],
        indicators: Dict[str, Any],
    ) -> PredictionExplanation:
        """
        Generate AI explanation for a prediction.

        Args:
            coin_id: e.g. "bitcoin"
            prediction: Dict with 'prediction', 'confidence', 'probability'
            indicators: Dict with RSI, MACD, Bollinger, volatility values

        Returns:
            PredictionExplanation Pydantic model
        """
        # Build prompt
        prompt = prediction_explanation_prompt(
            coin_id=coin_id,
            prediction=prediction.get('prediction', 'neutral'),
            confidence=prediction.get('confidence', 0.5),
            probability=prediction.get('probability', 0.5),
            indicators=indicators,
        )

        # Check cache
        cached = self.cache.get(prompt, "explain", coin_id)
        if cached:
            return PredictionExplanation(**cached)

        # Generate via LLM
        result = self.engine.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            schema_class=PredictionExplanation,
        )

        # Cache result
        self.cache.set(prompt, "explain", result, coin_id)

        return PredictionExplanation(**result)