from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

from src.predictor import CryptoPredictor
from src.model_manager import ModelManager
from src.database import CryptoDatabase

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="ML-powered Bitcoin price direction prediction",
    version="1.0.0"
)

# Global predictor instance (loaded once at startup)
predictor = None
db = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, db
    
    logger.info("Loading model...")
    db = CryptoDatabase()
    
    # Auto-load best model
    manager = ModelManager()
    model_path = manager.get_best_model('accuracy')
    
    if model_path and os.path.exists(model_path):
        predictor = CryptoPredictor()
        logger.info(f"API ready with model: {os.path.basename(model_path)}")
    else:
        logger.error("No model found!")
        predictor = None

# Request/Response models
class PredictionRequest(BaseModel):
    coin: str = "bitcoin"
    confidence_threshold: float = 0.6

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

# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """API info."""
    return {
        "name": "Crypto Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check."""
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
    - **confidence_threshold**: 0.0-1.0 (default 0.6)
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
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", tags=["Data"])
async def history(coin: str = "bitcoin", hours: int = 24):
    """Get recent price history."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        df = db.get_recent_prices(coin, hours=hours)
        return {
            "coin": coin,
            "records": len(df),
            "data": df.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
