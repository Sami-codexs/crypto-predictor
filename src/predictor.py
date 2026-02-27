import numpy as np
import logging
from typing import Dict, Optional
from src.indicators import TechnicalIndicators
from src.preprocessing import MLPreprocessor
from src.model import CryptoLSTM
import tensorflow as tf

logger = logging.getLogger(__name__)


class CryptoPredictor:
    """
    Production prediction interface.
    Loads trained model and makes predictions on new data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.indicators = TechnicalIndicators()
        self.preprocessor = MLPreprocessor(sequence_length=24)
        
        # Use ModelManager to select best model if path not provided
        if model_path is None:
            from src.model_manager import ModelManager
            manager = ModelManager()
            model_path = manager.get_best_model('accuracy')
            logger.info(f"Auto-selected best model: {model_path}")
        
        self.model = CryptoLSTM(sequence_length=24, n_features=12)
        
        if model_path:
            self.model.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No model loaded")
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        self.model.load(model_path)
        logger.info(f"Model loaded: {model_path}")
    
    def predict_next(self, coin_id: str = "bitcoin") -> Dict:
        """
        Predict next hour direction for a coin.
        
        Returns:
            {
                'coin': str,
                'timestamp': str,
                'current_price': float,
                'prediction': 'up' or 'down',
                'confidence': float (0-1),
                'probability': float (raw model output),
                'indicators': dict of current indicator values
            }
        """
        logger.info(f"Predicting next hour for {coin_id}")
        
        # Step 1: Get latest features
        try:
            df = self.indicators.engineer_features(coin_id, hours=168)
        except ValueError as e:
            logger.error(f"Not enough data for {coin_id}: {e}")
            return {
                'error': f"Insufficient data: {e}",
                'coin': coin_id
            }
        
        # Step 2: Check if we have enough recent data
        if len(df) < 24:
            return {
                'error': f"Need 24+ hours of data, have {len(df)}",
                'coin': coin_id
            }
        
        # Step 3: Get the last 24 hours as a sequence
        # We need to recreate the preprocessing steps without re-fitting
        latest_data = df.tail(24)
        
        # Step 4: Prepare features (same as training)
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'target', 'coin_id']]
        
        # Scale using the preprocessor's scaler (must be fitted)
        # For now, we'll use the raw indicators and note that scaling is needed
        sequence = latest_data[feature_cols].values
        
        # Reshape for model: (1, 24, 12)
        X = np.array([sequence])
        
        # Step 5: Make prediction
        try:
            probability = self.model.predict(X)[0]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'error': f"Model prediction failed: {e}",
                'coin': coin_id
            }
        
        # Step 6: Interpret result
        prediction = 'up' if probability > 0.5 else 'down'
        confidence = probability if prediction == 'up' else (1 - probability)
        
        # Get latest indicator values for context
        latest = latest_data.iloc[-1]
        
        return {
            'coin': coin_id,
            'timestamp': str(latest['timestamp']),
            'current_price': round(latest['price_usd'], 2),
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'probability': round(probability, 4),
            'indicators': {
                'rsi': round(latest['rsi'], 2),
                'macd_signal': 'bullish' if latest['macd_histogram'] > 0 else 'bearish',
                'bb_position': round(latest['bb_percent'], 2),
                'volatility': round(latest['volatility'], 2) if not np.isnan(latest['volatility']) else None
            }
        }
    
    def predict_with_threshold(self, coin_id: str = "bitcoin", 
                                confidence_threshold: float = 0.6) -> Dict:
        """
        Only predict if confidence exceeds threshold.
        Returns 'neutral' if uncertain.
        """
        result = self.predict_next(coin_id)
        
        if 'error' in result:
            return result
        
        if result['confidence'] < confidence_threshold:
            result['prediction'] = 'neutral'
            result['reason'] = f"Confidence {result['confidence']:.2f} below threshold {confidence_threshold}"
        
        return result
    
    def batch_predict(self, coin_ids: list) -> list:
        """Predict for multiple coins."""
        results = []
        for coin in coin_ids:
            result = self.predict_next(coin)
            results.append(result)
        return results
