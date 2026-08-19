import os
import numpy as np
import logging
import hashlib
import json
from typing import Dict, Optional, List
from datetime import datetime
from src.config import settings
from src.indicators import TechnicalIndicators
from src.preprocessing import MLPreprocessor
from src.model import CryptoLSTM
from src.database import CryptoDatabase

logger = logging.getLogger(__name__)


class CryptoPredictor:
    """
    Production prediction interface.
    Loads trained model, makes predictions on new data, and logs to DB.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        db: Optional[CryptoDatabase] = None,
    ):
        self.indicators = TechnicalIndicators()
        self.preprocessor = MLPreprocessor(sequence_length=settings.SEQUENCE_LENGTH)
        self.db = db or CryptoDatabase(settings.DB_PATH)

        if model_path is None:
            # Use the retrained model with correct feature count
            model_path = os.path.join('models', 'lstm_model.h5')
            scaler_path = os.path.join('models', 'scaler.pkl')
            if not os.path.exists(model_path):
                from src.model_manager import ModelManager
                manager = ModelManager()
                model_path = manager.get_best_model('accuracy')
                logger.info(f"Fallback to best model: {model_path}")
            else:
                logger.info(f"Using retrained model: {model_path}")

        self.model = CryptoLSTM(
            sequence_length=settings.SEQUENCE_LENGTH,
            n_features=12,
        )

        if model_path:
            self.model.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No model loaded")

        self.sequence_length = settings.SEQUENCE_LENGTH
        self.feature_cols = ['price_usd', 'volume_24h', 'rsi', 'macd_line', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower', 'bb_percent', 'price_change_1h', 'price_change_24h', 'volatility']

    def load_model(self, model_path: str):
        """Load a saved model."""
        self.model.load(model_path)
        logger.info(f"Model loaded: {model_path}")

    def predict_next(self, coin_id: str = "bitcoin") -> Dict:
        """
        Predict next hour direction for a coin.
        Logs to prediction_history and returns structured result.
        """
        logger.info(f"Predicting next hour for {coin_id}")

        try:
            df = self.indicators.engineer_features(coin_id, hours=168)
        except ValueError as e:
            logger.error(f"Not enough data for {coin_id}: {e}")
            return {
                'error': f"Insufficient data: {e}",
                'coin': coin_id,
            }

        if len(df) < self.sequence_length:
            return {
                'error': f"Need {self.sequence_length}+ hours of data, have {len(df)}",
                'coin': coin_id,
            }

        latest_data = df.tail(self.sequence_length)
        sequence = latest_data[self.feature_cols].values
        X = np.array([sequence])

        try:
            probability = float(self.model.predict(X)[0])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'error': f"Model prediction failed: {e}",
                'coin': coin_id,
            }

        prediction_class = 1 if probability > 0.5 else 0
        confidence = probability if prediction_class == 1 else (1 - probability)
        prediction_label = 'up' if prediction_class == 1 else 'down'

        latest = latest_data.iloc[-1]

        # ─── Build features hash for traceability ───
        features_hash = hashlib.sha256(
            json.dumps(sequence.tolist(), sort_keys=True).encode()
        ).hexdigest()[:16]

        result = {
            'coin': coin_id,
            'timestamp': str(latest['timestamp']),
            'current_price': float(round(float(latest['price_usd']), 2)),
            'prediction': prediction_label,
            'predicted_class': prediction_class,
            'confidence': float(round(confidence, 4)),
            'probability': float(round(probability, 4)),
            'features_hash': features_hash,
            'model_version': settings.MODEL_VERSION,
            'indicators': {
                'rsi': float(round(float(latest['rsi']), 2)),
                'macd_line': float(round(float(latest['macd_line']), 4)),
                'macd_signal': float(round(float(latest['macd_signal']), 4)),
                'macd_histogram': float(round(float(latest['macd_histogram']), 4)),
                'bb_upper': float(round(float(latest['bb_upper']), 2)),
                'bb_lower': float(round(float(latest['bb_lower']), 2)),
                'bb_percent': float(round(float(latest['bb_percent']), 2)),
                'price_change_1h': float(round(float(latest['price_change_1h']), 4)),
                'price_change_24h': float(round(float(latest['price_change_24h']), 4)),
                'volatility': (
                    float(round(float(latest['volatility']), 4))
                    if not np.isnan(latest['volatility'])
                    else None
                ),
            },
        }

        # ─── Persist to database ───
        try:
            pred_id = self.db.log_prediction(
                coin_id=coin_id,
                predicted_class=prediction_class,
                predicted_prob=probability,
                model_version=settings.MODEL_VERSION,
                features_hash=features_hash,
            )
            result['prediction_id'] = pred_id
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")

        return result

    def predict_with_threshold(
        self,
        coin_id: str = "bitcoin",
        confidence_threshold: float = 0.6,
    ) -> Dict:
        """
        Only predict if confidence exceeds threshold.
        Returns 'neutral' if uncertain.
        """
        result = self.predict_next(coin_id)

        if 'error' in result:
            return result

        conf = float(result['confidence'])
        if conf < confidence_threshold:
            result['prediction'] = 'neutral'
            result['reason'] = (
                f"Confidence {conf:.2f} below threshold {confidence_threshold}"
            )

        return result

    def batch_predict(self, coin_ids: List[str]) -> List[Dict]:
        """Predict for multiple coins."""
        return [self.predict_next(coin) for coin in coin_ids]

    def get_latest_indicators(self, coin_id: str = "bitcoin") -> Dict:
        """
        Return only the latest indicator snapshot (no prediction).
        Useful for the AI explainer layer.
        """
        try:
            df = self.indicators.engineer_features(coin_id, hours=168)
        except ValueError as e:
            return {'error': f"Insufficient data: {e}"}

        if len(df) < self.sequence_length:
            return {'error': f"Need {self.sequence_length}+ hours of data"}

        latest = df.iloc[-1]
        return {
            'coin': coin_id,
            'timestamp': str(latest['timestamp']),
            'price_usd': float(latest['price_usd']),
            'indicators': {
                col: (
                    float(latest[col])
                    if not np.isnan(latest[col])
                    else None
                )
                for col in self.feature_cols
            },
        }