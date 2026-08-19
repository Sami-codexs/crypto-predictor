"""
SQLite-based deduplication cache for LLM responses.
Identical prompts return cached responses, avoiding redundant Ollama calls.
"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from src.config import settings
from src.database import CryptoDatabase

logger = logging.getLogger(__name__)


class AICache:
    """
    Cache LLM responses by prompt hash to reduce API calls and latency.
    TTL configurable via settings.LLM_CACHE_TTL_HOURS.
    """
    
    def __init__(self, db: Optional[CryptoDatabase] = None):
        self.db = db or CryptoDatabase(settings.DB_PATH)
        self.ttl_hours = settings.LLM_CACHE_TTL_HOURS
    
    def _hash_prompt(self, prompt: str, prompt_type: str) -> str:
        """Create deterministic hash of prompt + type."""
        content = f"{prompt_type}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(
        self,
        prompt: str,
        prompt_type: str,
        coin_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if it exists and hasn't expired.
        
        Args:
            prompt: The full prompt text
            prompt_type: Category: 'explain', 'drift', 'briefing', 'chat'
            coin_id: Optional coin filter
        
        Returns:
            Parsed JSON dict if cache hit, None if miss or expired
        """
        cache_key = self._hash_prompt(prompt, prompt_type)
        
        try:
            cached = self.db.get_ai_cache(cache_key)
            if cached:
                logger.debug(f"Cache hit: {prompt_type} for {coin_id or 'general'}")
                return cached
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        
        return None
    
    def set(
        self,
        prompt: str,
        prompt_type: str,
        response: Dict[str, Any],
        coin_id: Optional[str] = None,
    ) -> bool:
        """
        Store response in cache.
        
        Args:
            prompt: The full prompt text
            prompt_type: Category: 'explain', 'drift', 'briefing', 'chat'
            response: Parsed JSON response dict
            coin_id: Optional coin filter
        
        Returns:
            True if stored successfully
        """
        cache_key = self._hash_prompt(prompt, prompt_type)
        
        try:
            response_json = json.dumps(response, sort_keys=True)
            self.db.set_ai_cache(
                cache_key=cache_key,
                prompt_type=prompt_type,
                response_json=response_json,
                coin_id=coin_id,
                ttl_hours=self.ttl_hours,
            )
            logger.debug(f"Cache set: {prompt_type} for {coin_id or 'general'}")
            return True
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
            return False
    
    def invalidate(
        self,
        prompt_type: Optional[str] = None,
        coin_id: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries. Use with caution.
        
        Args:
            prompt_type: If provided, only invalidate this type
            coin_id: If provided, only invalidate for this coin
        
        Returns:
            Number of entries invalidated
        """
        # For now, rely on TTL expiration. Manual invalidation
        # would require extending database.py with delete methods.
        logger.info(f"Cache invalidation requested: type={prompt_type}, coin={coin_id}")
        return 0
    
    def prune(self, max_age_days: int = 7) -> int:
        """
        Remove stale cache entries older than max_age_days.
        
        Returns:
            Number of entries pruned
        """
        try:
            self.db.prune_ai_cache(max_age_days=max_age_days)
            return 0  # db.prune_ai_cache doesn't return count currently
        except Exception as e:
            logger.warning(f"Cache prune failed: {e}")
            return 0