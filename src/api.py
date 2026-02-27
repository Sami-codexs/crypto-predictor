from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import logging
import os
import time
from datetime import datetime
from functools import lru_cache

from src.predictor import CryptoPredictor
from src.model_manager import ModelManager
from src.database import CryptoDatabase

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="ML-powered Bitcoin price direction prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global state
predictor = None
db = None
request_times = {}  # Simple rate limiting

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, db
    
    logger.info("Starting API...")
    db = CryptoDatabase()
    
    manager = ModelManager()
    model_path = manager.get_best_model('accuracy')
    
    if model_path and os.path.exists(model_path):
        predictor = CryptoPredictor()
        logger.info(f"API ready with model")
    else:
        logger.error("No model found!")

# Request/Response Models with Validation
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

class PredictionResponse(BaseModel):
    coin: str
    timestamp: str
    current_price: float
    prediction: str
    confidence: float
    probability: float
    indicators: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str
    uptime_seconds: Optional[int] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

# Rate limiting middleware
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Simple rate limiting: max 10 requests per minute per IP."""
    client_ip = request.client.host
    now = time.time()
    
    if client_ip not in request_times:
        request_times[client_ip] = []
    
    # Clean old requests (> 60 seconds)
    request_times[client_ip] = [t for t in request_times[client_ip] if now - t < 60]
    
    if len(request_times[client_ip]) >= 10:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": 60}
        )
    
    request_times[client_ip].append(now)
    response = await call_next(request)
    return response

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """API info."""
    return {
        "name": "Crypto Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)",
            "history": "/history?coin=bitcoin&hours=24"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check with detailed status."""
    return {
        "status": "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "database_connected": db is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict next hour price direction.
    
    - **coin**: bitcoin or ethereum
    - **confidence_threshold**: 0.0-1.0, default 0.6
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predictor.predict_with_threshold(
            coin_id=request.coin,
            confidence_threshold=request.confidence_threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/history", tags=["Data"])
async def history(coin: str = "bitcoin", hours: int = 24):
    """
    Get recent price history.
    
    - **coin**: bitcoin or ethereum
    - **hours**: 1-168 (default 24)
    """
    # Validate hours
    if not 1 <= hours <= 168:
        raise HTTPException(status_code=400, detail="Hours must be 1-168")
    
    # Validate coin
    allowed = ['bitcoin', 'ethereum', 'btc', 'eth']
    if coin.lower() not in allowed:
        raise HTTPException(status_code=400, detail=f"Coin must be one of: {allowed}")
    
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        df = db.get_recent_prices(coin, hours=hours)
        return {
            "coin": coin,
            "hours": hours,
            "records": len(df),
            "latest_price": float(df['price_usd'].iloc[0]) if len(df) > 0 else None,
            "data": df.head(10).to_dict('records')  # Limit to 10 records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)