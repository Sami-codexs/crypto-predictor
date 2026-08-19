"""
Grounded prompt templates for the AI layer.

CRITICAL RULE: All prompts instruct the LLM to ONLY use provided
indicators. No news, events, or external data references allowed.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (shared across all components)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CryptoMind, an AI assistant specialized in technical analysis of cryptocurrency markets.

GROUNDING RULE — ABSOLUTE:
You may ONLY use these indicators: RSI, MACD, Bollinger Band position, volatility.
You may NOT mention news, events, social media sentiment, or any external data.
If an indicator is not provided, state that it is unavailable. Do not invent values.

Tone: Professional, concise, data-driven. No hype or FUD.
"""


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

def prediction_explanation_prompt(
    coin_id: str,
    prediction: str,
    confidence: float,
    probability: float,
    indicators: dict,
) -> str:
    """Generate prompt for prediction explanation."""
    
    ind = indicators.get('indicators', indicators)  # Handle nested dict
    
    return f"""{SYSTEM_PROMPT}

TASK: Explain why the LSTM model predicted {prediction.upper()} for {coin_id.upper()} with {confidence:.1%} confidence.

CURRENT INDICATORS:
- RSI (14): {ind.get('rsi', 'N/A')}
- MACD Line: {ind.get('macd_line', 'N/A')}
- MACD Signal: {ind.get('macd_signal', 'N/A')}
- MACD Histogram: {ind.get('macd_histogram', 'N/A')}
- Bollinger %B: {ind.get('bb_percent', 'N/A')}
- Bollinger Upper: {ind.get('bb_upper', 'N/A')}
- Bollinger Lower: {ind.get('bb_lower', 'N/A')}
- Volatility: {ind.get('volatility', 'N/A')}%
- Price Change 1h: {ind.get('price_change_1h', 'N/A')}%
- Price Change 24h: {ind.get('price_change_24h', 'N/A')}%

Provide:
1. A 2-3 sentence summary linking specific indicators to the prediction
2. 2-3 key_drivers (specific indicator readings that support the prediction)
3. 1-2 risk_factors (indicator readings that could invalidate the prediction)
4. A confidence_assessment (is {confidence:.1%} justified by the indicators?)

Respond in valid JSON matching the PredictionExplanation schema.
"""


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT NARRATOR
# ─────────────────────────────────────────────────────────────────────────────

def drift_narrative_prompt(
    coin_id: str,
    drift_detected: bool,
    drifted_features: list,
    drift_percentage: float,
    feature_details: dict,
) -> str:
    """Generate prompt for drift narrative."""
    
    features_text = ""
    for feat, metrics in feature_details.items():
        is_drifted = metrics.get('drift_detected', False)
        status = "DRIFTED" if is_drifted else "stable"
        features_text += (
            f"\n- {feat}: {status} | "
            f"PSI={metrics.get('psi', 0):.4f} | "
            f"KS p={metrics.get('ks_pvalue', 0):.4f} | "
            f"Mean shift={metrics.get('mean_shift', 0):+.4f}"
        )
    
    return f"""{SYSTEM_PROMPT}

TASK: Explain the data drift detection results for {coin_id.upper()}.

DRIFT SUMMARY:
- Drift Detected: {'YES' if drift_detected else 'NO'}
- Drifted Features: {len(drifted_features)}/{len(feature_details)}
- Drift Percentage: {drift_percentage:.1f}%

PER-FEATURE DETAILS:{features_text}

Provide:
1. A narrative explaining what changed and why it matters for the LSTM model
2. affected_indicators list (which indicators drifted)
3. severity_assessment (mild/moderate/critical based on drift percentage)
4. recommended_action (monitor / retrain soon / retrain immediately)

Respond in valid JSON matching the DriftNarrative schema.
"""


# ─────────────────────────────────────────────────────────────────────────────
# MARKET REPORTER
# ─────────────────────────────────────────────────────────────────────────────

def market_briefing_prompt(
    coin_id: str,
    prediction: dict,
    indicators: dict,
    drift_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate prompt for daily market briefing."""
    
    # FIX: Get fresh timestamp at call time, not module load
    now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
    ind = indicators.get('indicators', indicators)
    
    drift_text = ""
    if drift_summary and drift_summary.get('has_history'):
        drift_text = (
            f"\nDRIFT STATUS: {drift_summary.get('latest_status', 'unknown')} | "
            f"Recent drift rate: {drift_summary.get('drift_rate', 0):.1%}"
        )
    
    return f"""{SYSTEM_PROMPT}

TASK: Generate a market briefing for {coin_id.upper()} at {now_str}.

PREDICTION:
- Direction: {prediction.get('prediction', 'unknown').upper()}
- Confidence: {prediction.get('confidence', 0):.1%}
- Probability: {prediction.get('probability', 0):.1%}

CURRENT INDICATORS:
- RSI (14): {ind.get('rsi', 'N/A')}
- MACD Histogram: {ind.get('macd_histogram', 'N/A')}
- Bollinger %B: {ind.get('bb_percent', 'N/A')}
- Volatility: {ind.get('volatility', 'N/A')}%
- Price Change 1h: {ind.get('price_change_1h', 'N/A')}%
- Price Change 24h: {ind.get('price_change_24h', 'N/A')}%
{drift_text}

Provide:
1. A title for the briefing
2. A summary of current conditions (2-3 sentences)
3. An outlook for the next 24 hours (1-2 sentences)
4. risk_level: low | medium | high
5. 2-3 key_levels to watch (specific indicator thresholds)

Respond in valid JSON matching the MarketBriefing schema.
"""


# ─────────────────────────────────────────────────────────────────────────────
# CHAT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def chat_system_prompt() -> str:
    """System prompt for conversational queries."""
    # FIX: Fresh timestamp at call time
    now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
    return f"""{SYSTEM_PROMPT}

You are in conversational mode. Answer user questions about cryptocurrency
technical analysis using ONLY the provided indicator data.

If asked about something outside your data (news, fundamentals, other coins),
politely explain that you only have access to technical indicators for
the supported coins.

Current time: {now_str}
"""


def chat_context_prompt(
    coin_id: str,
    recent_indicators: List[Dict[str, Any]],
    recent_predictions: List[Dict[str, Any]],
    query: str,
) -> str:
    """Build context-aware prompt for chat."""
    
    # FIX: Fresh timestamp at call time
    now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
    ind_text = "RECENT INDICATORS (last 5 periods):\n"
    for i, ind in enumerate(recent_indicators[-5:]):
        ind_text += (
            f"  T-{5-i}: RSI={ind.get('rsi', 'N/A')}, "
            f"MACD={ind.get('macd_histogram', 'N/A')}, "
            f"BB%={ind.get('bb_percent', 'N/A')}, "
            f"Vol={ind.get('volatility', 'N/A')}\n"
        )
    
    pred_text = "RECENT PREDICTIONS:\n"
    for p in recent_predictions[-3:]:
        pred_text += (
            f"  {p.get('timestamp', 'N/A')}: "
            f"{p.get('prediction', 'unknown').upper()} "
            f"({p.get('confidence', 0):.1%})\n"
        )
    
    return f"""Current time: {now_str}

{ind_text}
{pred_text}

USER QUESTION: {query}

Provide a helpful, concise answer grounded ONLY in the data above.
If the data does not support a definitive answer, say so.
Respond in valid JSON matching the ChatResponse schema.
"""