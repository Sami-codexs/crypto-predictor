"""
Pydantic schemas for structured LLM output.
All AI components return these models for type-safe consumption.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class PredictionExplanation(BaseModel):
    """AI-generated narrative explaining a prediction."""
    
    summary: str = Field(
        ...,
        description="2-3 sentence explanation of why the model predicted up/down",
        min_length=10,
        max_length=500,
    )
    key_drivers: List[str] = Field(
        default_factory=list,
        description="List of specific indicator-based reasons for the prediction",
        max_length=5,
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Indicator-based risks that could invalidate the prediction",
        max_length=5,
    )
    confidence_assessment: str = Field(
        default="",
        description="Assessment of whether the confidence score is justified by indicators",
        max_length=200,
    )


class DriftNarrative(BaseModel):
    """AI-generated narrative explaining data drift results."""
    
    narrative: str = Field(
        ...,
        description="Human-readable explanation of what drifted and why it matters",
        min_length=10,
        max_length=800,
    )
    affected_indicators: List[str] = Field(
        default_factory=list,
        description="Which specific indicators are drifting",
    )
    severity_assessment: str = Field(
        default="",
        description="Whether the drift is mild, moderate, or critical",
        max_length=200,
    )
    recommended_action: str = Field(
        default="",
        description="Specific action: monitor, retrain soon, or retrain immediately",
        max_length=200,
    )


class MarketBriefing(BaseModel):
    """AI-generated daily market briefing."""
    
    title: str = Field(
        default="Market Briefing",
        description="Briefing headline",
        max_length=100,
    )
    summary: str = Field(
        ...,
        description="Executive summary of current market conditions based on indicators",
        min_length=20,
        max_length=600,
    )
    outlook: str = Field(
        ...,
        description="Forward-looking assessment for next 24h based on indicator trends",
        min_length=10,
        max_length=400,
    )
    risk_level: str = Field(
        default="medium",
        description="low | medium | high",
        pattern="^(low|medium|high|unknown)$",
    )
    key_levels: List[str] = Field(
        default_factory=list,
        description="Important indicator thresholds to watch",
        max_length=5,
    )


class ChatResponse(BaseModel):
    """AI response to conversational query."""
    
    response: str = Field(
        ...,
        description="Natural language answer grounded in available data",
        min_length=1,
        max_length=1000,
    )
    data_used: List[str] = Field(
        default_factory=list,
        description="Which data sources were referenced (indicators, predictions, history)",
    )
    confidence: str = Field(
        default="medium",
        description="How certain the answer is based on available data",
        pattern="^(high|medium|low|uncertain)$",
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="Follow-up suggestion for the user",
        max_length=200,
    )