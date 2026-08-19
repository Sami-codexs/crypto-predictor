from datetime import datetime
"""
DriftNarrator: Translates statistical drift detection results into
human-readable English narratives.
"""

import logging
from typing import Dict, Optional, Any

from src.ai.engine import AIEngine
from src.ai.cache import AICache
from src.ai.prompts import drift_narrative_prompt, SYSTEM_PROMPT
from src.ai.schemas import DriftNarrative

logger = logging.getLogger(__name__)


class DriftNarrator:
    """
    Generates natural language explanations of data drift.
    Helps non-technical users understand why model retraining is needed.
    """

    def __init__(self, engine: Optional[AIEngine] = None, cache: Optional[AICache] = None):
        self.engine = engine or AIEngine()
        self.cache = cache or AICache()

    def narrate(self, drift_results: Dict[str, Any]) -> DriftNarrative:
        """
        Generate AI narrative for drift detection results.

        Args:
            drift_results: Output from DriftDetector.detect_drift()

        Returns:
            DriftNarrative Pydantic model
        """
        # Extract required fields with safe defaults
        features = drift_results.get('features', {})
        drifted = drift_results.get('drifted_features', [])
        detected = drift_results.get('drift_detected', False)
        pct = drift_results.get('drift_percentage', 0.0)

        # Build prompt
        prompt = drift_narrative_prompt(
            coin_id=drift_results.get('coin_id', 'unknown'),
            drift_detected=detected,
            drifted_features=drifted,
            drift_percentage=pct,
            feature_details=features,
        )

        # Check cache (drift results are less cacheable due to changing stats,
        # but we cache for short TTL)
        cached = self.cache.get(prompt, "drift", None)
        if cached:
            return DriftNarrative(**cached)

        # Generate via LLM
        result = self.engine.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            schema_class=DriftNarrative,
        )

        # Cache result
        self.cache.set(prompt, "drift", result, None)

        if isinstance(result, DriftNarrative):
            return result
        return DriftNarrative(**result)