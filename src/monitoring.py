"""
Day 22: Production Monitoring

Exposes Prometheus-style metrics for ML model monitoring.
Tracks predictions, latency, confidence scores, and system health.

Metrics Endpoint: GET /metrics

References:
- https://www.jeremyjordan.me/ml-monitoring/
- https://prometheus.io/docs/practices/naming/
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time
import logging
from typing import Callable
from fastapi import Response

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Production monitoring for ML model.
    Tracks predictions, latency, and business metrics.
    """
    
    def __init__(self):
        # Prediction metrics
        self.prediction_counter = Counter(
            'crypto_predictions_total',
            'Total number of predictions made',
            ['coin', 'prediction_type']  # Labels: bitcoin/up, bitcoin/down, etc.
        )
        
        # Confidence score distribution
        self.confidence_histogram = Histogram(
            'crypto_prediction_confidence',
            'Distribution of prediction confidence scores',
            ['coin'],
            buckets=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        )
        
        # Latency metrics
        self.prediction_latency = Histogram(
            'crypto_prediction_latency_seconds',
            'Time spent processing prediction requests',
            ['endpoint'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        self.data_fetch_latency = Histogram(
            'crypto_data_fetch_latency_seconds',
            'Time spent fetching data from database',
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'crypto_model_accuracy',
            'Current model accuracy from backtesting',
            ['model_name']
        )
        
        self.model_sharpe = Gauge(
            'crypto_model_sharpe_ratio',
            'Sharpe ratio from backtesting',
            ['model_name']
        )
        
        # System metrics
        self.active_requests = Gauge(
            'crypto_active_requests',
            'Number of requests currently being processed'
        )
        
        self.db_records = Gauge(
            'crypto_database_records',
            'Number of price records in database',
            ['coin']
        )
        
        # Model info
        self.model_info = Info(
            'crypto_model',
            'Model metadata'
        )
    
    def track_prediction(self, coin: str, prediction: str, confidence: float):
        """Track a prediction with its confidence."""
        self.prediction_counter.labels(
            coin=coin, 
            prediction_type=prediction
        ).inc()
        
        self.confidence_histogram.labels(coin=coin).observe(confidence)
        
        logger.debug(f"Tracked prediction: {coin}={prediction} (conf={confidence:.3f})")
    
    def track_latency(self, endpoint: str, duration: float):
        """Track request latency."""
        self.prediction_latency.labels(endpoint=endpoint).observe(duration)
    
    def track_data_fetch(self, duration: float):
        """Track database fetch latency."""
        self.data_fetch_latency.observe(duration)
    
    def update_model_metrics(self, model_name: str, accuracy: float, sharpe: float):
        """Update model performance metrics."""
        self.model_accuracy.labels(model_name=model_name).set(accuracy)
        self.model_sharpe.labels(model_name=model_name).set(sharpe)
    
    def update_db_count(self, coin: str, count: int):
        """Update database record count."""
        self.db_records.labels(coin=coin).set(count)
    
    def set_model_info(self, model_name: str, version: str, training_date: str):
        """Set model metadata."""
        self.model_info.info({
            'model_name': model_name,
            'version': version,
            'training_date': training_date
        })
    
    def timer(self, metric_name: str):
        """Context manager for timing operations."""
        return TimerContextManager(self, metric_name)
    
    def get_metrics(self) -> bytes:
        """Generate Prometheus format metrics."""
        return generate_latest()


class TimerContextManager:
    """Context manager for timing code blocks."""
    
    def __init__(self, monitor: ModelMonitor, metric_name: str):
        self.monitor = monitor
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.metric_name == 'data_fetch':
            self.monitor.track_data_fetch(duration)
        else:
            self.monitor.track_latency(self.metric_name, duration)


# Global monitor instance
# Global monitor instance - created lazily
_monitor_instance = None

def get_monitor() -> ModelMonitor:
    """Get or create global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance

# For testing - reset global instance
def _reset_monitor():
    """Reset global monitor (for testing only)."""
    global _monitor_instance
    _monitor_instance = None

# For FastAPI integration
def setup_monitoring(app):
    """
    Setup monitoring endpoints for FastAPI app.
    
    Usage:
        from fastapi import FastAPI
        from monitoring import setup_monitoring
        
        app = FastAPI()
        setup_monitoring(app)
    """
    from fastapi import Response
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        monitor = get_monitor()
        return Response(
            content=monitor.get_metrics(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/health")
    async def health():
        """Health check endpoint with metrics."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "monitoring": "enabled"
        }
    
    logger.info("Monitoring setup complete: /metrics endpoint available")


if __name__ == "__main__":
    # Demo usage
    monitor = ModelMonitor()
    
    print("Day 22: Production Monitoring")
    print("=" * 50)
    
    # Simulate some predictions
    for i in range(10):
        coin = "bitcoin" if i % 2 == 0 else "ethereum"
        pred = "up" if i % 3 == 0 else "down"
        conf = 0.5 + (i * 0.05)
        
        monitor.track_prediction(coin, pred, conf)
    
    # Update model metrics
    monitor.update_model_metrics("lstm_v1", accuracy=0.62, sharpe=1.2)
    
    # Generate metrics output
    metrics_output = monitor.get_metrics()
    print("\nMetrics Output (first 1000 chars):")
    print(metrics_output[:1000].decode('utf-8'))