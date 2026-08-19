from fastapi import FastAPI, HTTPException, Request, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
import os
import time
import json
from datetime import datetime
import numpy as np

from src.backtester import Backtester
from src.predictor import CryptoPredictor
from src.model_manager import ModelManager
from src.database import CryptoDatabase
from src.config import settings
from src.drift_detector import DriftDetector

# ─── AI Layer imports (lazy to avoid startup crash if not yet present) ───
_ai_imported = False
_ai_engine = None
_ai_explainer = None
_ai_drift_narrator = None
_ai_reporter = None
_ai_chat = None

def _ensure_ai():
    global _ai_imported, _ai_engine, _ai_explainer, _ai_drift_narrator, _ai_reporter, _ai_chat
    if _ai_imported:
        return
    try:
        from src.ai.engine import AIEngine
        from src.ai.explainer import PredictionExplainer
        from src.ai.drift_narrator import DriftNarrator
        from src.ai.reporter import MarketReporter
        from src.ai.chat import ChatEngine

        _ai_engine = AIEngine()
        _ai_explainer = PredictionExplainer(_ai_engine)
        _ai_drift_narrator = DriftNarrator(_ai_engine)
        _ai_reporter = MarketReporter(_ai_engine)
        _ai_chat = ChatEngine(_ai_engine)
        _ai_imported = True
        logger.info("AI layer loaded successfully")
    except Exception as e:
        logger.warning(f"AI layer not available: {e}")
        _ai_imported = True  # Don't retry

logger = logging.getLogger(__name__)


def convert_numpy(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(i) for i in obj)
    return obj


# ─── Request/Response Models ───

class PredictionRequest(BaseModel):
    coin: str = Field(default="bitcoin", min_length=3, max_length=20)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    @validator('coin')
    def validate_coin(cls, v):
        allowed = ['bitcoin', 'ethereum', 'btc', 'eth']
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"Coin must be one of: {allowed}")
        return v


class BatchPredictionRequest(BaseModel):
    coins: List[str] = Field(default=["bitcoin", "ethereum"], min_items=1, max_items=10)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    coin: Optional[str] = Field(default="bitcoin")
    context_hours: int = Field(default=24, ge=1, le=168)


class PredictionResponse(BaseModel):
    coin: str
    timestamp: str
    current_price: float
    prediction: str
    confidence: float
    probability: float
    indicators: dict


class PerformanceResponse(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    last_updated: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    ai_available: bool
    timestamp: str
    uptime_seconds: Optional[int] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str


# ─── FastAPI App ───

app = FastAPI(
    title="CryptoMind API",
    description="ML-powered crypto price direction prediction with AI intelligence layer",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

predictor = None
db = None
drift_detector = None
request_times = {}

_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Load model and AI layer on startup."""
    global predictor, db, drift_detector

    logger.info("Starting CryptoMind API...")
    db = CryptoDatabase(settings.DB_PATH)

    manager = ModelManager()
    model_path = manager.get_best_model('accuracy')

    if model_path and os.path.exists(model_path):
        predictor = CryptoPredictor(model_path=model_path, db=db)
        logger.info("Predictor ready")
    else:
        logger.error("No model found!")

    # Initialize drift detector with recent reference data
    try:
        ref_df = db.get_indicator_window(coin_id="bitcoin", lookback=settings.DRIFT_WINDOW_SIZE)
        if len(ref_df) >= settings.DRIFT_WINDOW_SIZE:
            drift_detector = DriftDetector(reference_data=ref_df, db=db)
            logger.info("Drift detector initialized")
    except Exception as e:
        logger.warning(f"Drift detector init failed: {e}")

    # Lazy-load AI layer
    _ensure_ai()


# ─── Middleware ───

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Simple rate limiting: max 60 requests per minute per IP."""
    client_ip = request.client.host
    now = time.time()

    if client_ip not in request_times:
        request_times[client_ip] = []

    request_times[client_ip] = [t for t in request_times[client_ip] if now - t < 60]

    if len(request_times[client_ip]) >= 60:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": 60},
        )

    request_times[client_ip].append(now)
    response = await call_next(request)
    return response


# ─── Exception Handler ───

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=convert_numpy({
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        }),
    )


# ─── Root & Health ───

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "CryptoMind API",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_explain": "/predict/explain (POST)",
            "drift": "/drift?coin=bitcoin",
            "briefing": "/briefing?coin=bitcoin",
            "chat": "/chat (POST)",
            "history": "/history?coin=bitcoin&hours=24",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return {
        "status": "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "database_connected": db is not None,
        "ai_available": _ai_imported and _ai_engine is not None,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - _start_time),
    }


# ─── Prediction Endpoints ───

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.predict_with_threshold(
            coin_id=request.coin,
            confidence_threshold=request.confidence_threshold,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return convert_numpy(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/explain", tags=["Prediction", "AI"])
async def predict_explain(request: PredictionRequest):
    """
    Return prediction + AI-generated natural language explanation
    grounded ONLY in the provided technical indicators.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    _ensure_ai()
    if _ai_explainer is None:
        raise HTTPException(status_code=503, detail="AI layer not available")

    try:
        # 1. Get prediction
        pred = predictor.predict_with_threshold(
            coin_id=request.coin,
            confidence_threshold=request.confidence_threshold,
        )
        if "error" in pred:
            raise HTTPException(status_code=400, detail=pred["error"])

        # 2. Get indicator snapshot
        indicators = predictor.get_latest_indicators(request.coin)
        if "error" in indicators:
            raise HTTPException(status_code=400, detail=indicators["error"])

        # 3. Generate explanation
        explanation = _ai_explainer.explain(
            coin_id=request.coin,
            prediction=pred,
            indicators=indicators,
        )

        return convert_numpy({
            "prediction": pred,
            "explanation": explanation.dict() if hasattr(explanation, "dict") else explanation,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explain failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    allowed = ['bitcoin', 'ethereum', 'btc', 'eth']
    invalid_coins = [c for c in request.coins if c.lower() not in allowed]
    if invalid_coins:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid coins: {invalid_coins}. Allowed: {allowed}",
        )

    results = []
    up_count = down_count = neutral_count = 0

    for coin in request.coins:
        try:
            result = predictor.predict_with_threshold(
                coin_id=coin,
                confidence_threshold=request.confidence_threshold,
            )
            results.append(result)
            pred = result.get('prediction', 'neutral')
            if pred == 'up':
                up_count += 1
            elif pred == 'down':
                down_count += 1
            else:
                neutral_count += 1
        except Exception as e:
            results.append({'coin': coin, 'error': str(e)})

    return convert_numpy({
        "predictions": results,
        "summary": {
            "total": len(request.coins),
            "up": up_count,
            "down": down_count,
            "neutral": neutral_count,
            "threshold": request.confidence_threshold,
        },
    })


# ─── Drift Endpoint ───

@app.get("/drift", tags=["Monitoring", "AI"])
async def drift_check(
    coin: str = Query(default="bitcoin"),
    hours: int = Query(default=48, ge=12, le=168),
):
    """
    Run drift detection and return structured stats + AI narrative.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")

    allowed = ['bitcoin', 'ethereum', 'btc', 'eth']
    if coin.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"Coin must be one of: {allowed}")

    try:
        df = predictor.indicators.engineer_features(coin, hours=hours)
        feature_cols = [
            c for c in df.columns
            if c not in ['timestamp', 'target', 'coin_id']
        ]
        current_features = df[feature_cols].dropna()

        if len(current_features) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for drift check")

        results = drift_detector.detect_drift(current_features)

        # AI narrative
        narrative = None
        _ensure_ai()
        if _ai_drift_narrator:
            narrative = _ai_drift_narrator.narrate(results)

        # Recent summary from DB
        summary = drift_detector.get_drift_summary(coin_id=coin, limit=5)

        return convert_numpy({
            "drift_detected": results['drift_detected'],
            "drifted_features": results['drifted_features'],
            "drift_percentage": results.get('drift_percentage', 0),
            "recommendation": results['recommendation'],
            "feature_details": results['features'],
            "narrative": narrative.dict() if hasattr(narrative, "dict") else narrative,
            "recent_summary": summary,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Briefing Endpoint ───

@app.get("/briefing", tags=["AI"])
async def market_briefing(
    coin: str = Query(default="bitcoin"),
    hours: int = Query(default=24, ge=6, le=168),
):
    """
    Generate an AI-powered daily market briefing grounded in technical indicators.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _ensure_ai()
    if _ai_reporter is None:
        raise HTTPException(status_code=503, detail="AI layer not available")

    allowed = ['bitcoin', 'ethereum', 'btc', 'eth']
    if coin.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"Coin must be one of: {allowed}")

    try:
        # Get prediction + indicators
        pred = predictor.predict_next(coin)
        if "error" in pred:
            raise HTTPException(status_code=400, detail=pred["error"])

        indicators = predictor.get_latest_indicators(coin)
        if "error" in indicators:
            raise HTTPException(status_code=400, detail=indicators["error"])

        # Get recent drift summary
        drift_summary = None
        if drift_detector:
            drift_summary = drift_detector.get_drift_summary(coin_id=coin, limit=5)

        briefing = _ai_reporter.generate(
            coin_id=coin,
            prediction=pred,
            indicators=indicators,
            drift_summary=drift_summary,
        )

        return convert_numpy({
            "coin": coin,
            "timestamp": datetime.now().isoformat(),
            "briefing": briefing.dict() if hasattr(briefing, "dict") else briefing,
            "raw_prediction": pred,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Briefing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chat Endpoint ───

@app.post("/chat", tags=["AI"])
async def chat(request: ChatRequest):
    """
    Conversational query engine with DB retrieval.
    """
    _ensure_ai()
    if _ai_chat is None:
        raise HTTPException(status_code=503, detail="AI layer not available")

    allowed = ['bitcoin', 'ethereum', 'btc', 'eth']
    if request.coin and request.coin.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"Coin must be one of: {allowed}")

    try:
        answer = _ai_chat.ask(
            query=request.query,
            coin_id=request.coin,
            context_hours=request.context_hours,
            db=db,
        )
        return convert_numpy({
            "query": request.query,
            "coin": request.coin,
            "answer": answer.dict() if hasattr(answer, "dict") else answer,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── History & Performance ───

@app.get("/history", tags=["Data"])
async def history(coin: str = "bitcoin", hours: int = 24):
    if not 1 <= hours <= 168:
        raise HTTPException(status_code=400, detail="Hours must be 1-168")

    allowed = ['bitcoin', 'ethereum', 'btc', 'eth']
    if coin.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"Coin must be one of: {allowed}")

    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        df = db.get_recent_prices(coin, hours=hours)
        return convert_numpy({
            "coin": coin,
            "hours": hours,
            "records": len(df),
            "latest_price": float(df['price_usd'].iloc[0]) if len(df) > 0 else None,
            "data": df.head(10).to_dict('records'),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/performance", response_model=PerformanceResponse, tags=["Performance"])
async def performance(coin: str = "bitcoin", days: int = 7):
    if not 1 <= days <= 30:
        raise HTTPException(status_code=400, detail="Days must be 1-30")

    allowed = ['bitcoin', 'ethereum', 'btc', 'eth']
    if coin.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"Coin must be one of: {allowed}")

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        hours = days * 24
        df = predictor.indicators.engineer_features(coin, hours=hours)

        if len(df) < 48:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: need 48+ hours, have {len(df)}",
            )

        backtester = Backtester(initial_capital=10000.0)
        metrics = backtester.run_backtest(
            df, predictor.model, threshold=0.53, fee_pct=0.001
        )

        manager = ModelManager()
        models = manager.list_models()
        model_name = models[0]['name'] if models else "unknown"

        # Handle nested metrics structure from updated backtester
        classification = metrics.get('classification', {})
        trading = metrics.get('trading', {})

        result = {
            "model_name": model_name,
            "accuracy": float(classification.get('accuracy', 0)),
            "precision": float(classification.get('precision', 0)),
            "recall": float(classification.get('recall', 0)),
            "f1_score": float(classification.get('f1_score', 0)),
            "total_trades": int(trading.get('total_trades', 0)),
            "win_rate": float(trading.get('win_rate', 0)),
            "total_return_pct": float(trading.get('total_return_pct', 0)),
            "max_drawdown_pct": float(trading.get('max_drawdown_pct', 0)),
            "sharpe_ratio": float(trading.get('sharpe_ratio', 0)),
            "last_updated": datetime.now().isoformat(),
        }

        return convert_numpy(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/history", tags=["History"])
async def predictions_history(coin: str = "bitcoin", limit: int = 100):
    """
    Get recent prediction history from the database.
    """
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=400, detail="Limit must be 1-1000")

    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        df = db.get_predictions(coin_id=coin, limit=limit)
        return convert_numpy({
            "coin": coin,
            "total": len(df),
            "predictions": df.to_dict('records') if not df.empty else [],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/compare", tags=["Models"])
async def compare_models():
    try:
        manager = ModelManager()
        models = manager.list_models()

        if not models:
            return {"error": "No models found"}

        comparison = []
        for model in models[:5]:
            info = {
                "name": model['name'],
                "created": model.get('created', 'unknown'),
                "accuracy": (
                    float(model.get('accuracy'))
                    if model.get('accuracy') is not None
                    else None
                ),
            }
            comparison.append(info)

        return convert_numpy({
            "total_models": len(models),
            "models": comparison,
            "best_model": models[0]['name'] if models else None,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT)