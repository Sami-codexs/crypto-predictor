import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import sqlite3

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages multiple model versions, tracks performance, handles rollbacks.
    """
    
    def __init__(self, models_dir: str = "models", db_path: str = "data/crypto.db"):
        self.models_dir = models_dir
        self.db_path = db_path
        os.makedirs(models_dir, exist_ok=True)
        self.init_performance_table()
    
    def init_performance_table(self):
        """Track model performance over time."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    test_samples INTEGER,
                    notes TEXT
                )
            """)
            conn.commit()
    
    def list_models(self) -> List[Dict]:
        """List all available models with metadata."""
        models = []
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.keras'):
                model_path = os.path.join(self.models_dir, filename)
                meta_path = model_path.replace('.keras', '_metadata.json')
                
                model_info = {
                    'name': filename,
                    'path': model_path,
                    'created': None,
                    'accuracy': None
                }
                
                # Load metadata if exists
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            model_info['created'] = meta.get('timestamp')
                            if 'history' in meta and 'val_accuracy' in meta['history']:
                                model_info['accuracy'] = meta['history']['val_accuracy'][-1]
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {filename}: {e}")
                
                models.append(model_info)
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x['created'] or '', reverse=True)
        return models
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[str]:
        """
        Select best model by metric.
        Returns path to best model.
        """
        models = self.list_models()
        
        if not models:
            logger.error("No models found")
            return None
        
        # Filter models with accuracy data
        valid_models = [m for m in models if m.get(metric) is not None]
        
        if not valid_models:
            # Fallback: return most recent
            logger.warning(f"No models with {metric} data, using most recent")
            return models[0]['path']
        
        # Sort by metric (higher is better)
        best = max(valid_models, key=lambda x: x[metric])
        logger.info(f"Best model by {metric}: {best['name']} ({best[metric]:.3f})")
        
        return best['path']
    
    def get_latest_model(self) -> Optional[str]:
        """Get most recently created model."""
        models = self.list_models()
        if not models:
            return None
        return models[0]['path']
    
    def log_performance(self, model_name: str, metrics: Dict, notes: str = ""):
        """Log model performance to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_performance 
                (model_name, accuracy, precision, recall, f1_score, test_samples, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                metrics.get('accuracy'),
                metrics.get('precision'),
                metrics.get('recall'),
                metrics.get('f1_score'),
                metrics.get('test_samples', 0),
                notes
            ))
            conn.commit()
        
        logger.info(f"Logged performance for {model_name}: acc={metrics.get('accuracy', 0):.3f}")
    
    def get_performance_history(self, model_name: Optional[str] = None) -> List[Dict]:
        """Get performance history for a model or all models."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_name:
                cursor.execute("""
                    SELECT * FROM model_performance 
                    WHERE model_name = ? 
                    ORDER BY timestamp DESC
                """, (model_name,))
            else:
                cursor.execute("""
                    SELECT * FROM model_performance 
                    ORDER BY timestamp DESC
                """)
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
    
    def cleanup_old_models(self, keep_latest: int = 5):
        """Delete old model files, keep only N most recent."""
        models = self.list_models()
        
        if len(models) <= keep_latest:
            return
        
        to_delete = models[keep_latest:]
        
        for model in to_delete:
            try:
                os.remove(model['path'])
                # Also delete metadata and scaler if exist
                for ext in ['_metadata.json', '_scaler.pkl']:
                    extra_file = model['path'].replace('.keras', ext)
                    if os.path.exists(extra_file):
                        os.remove(extra_file)
                logger.info(f"Deleted old model: {model['name']}")
            except Exception as e:
                logger.error(f"Failed to delete {model['name']}: {e}")
    
    def compare_models(self, model_names: List[str]) -> str:
        """Generate comparison report of multiple models."""
        report = ["Model Comparison Report", "=" * 50, ""]
        
        for name in model_names:
            history = self.get_performance_history(name)
            if history:
                latest = history[0]
                report.append(f"{name}:")
                report.append(f"  Accuracy:  {latest['accuracy']:.3f}")
                report.append(f"  Precision: {latest['precision']:.3f}")
                report.append(f"  Recall:    {latest['recall']:.3f}")
                report.append(f"  F1:        {latest['f1_score']:.3f}")
                report.append(f"  Tested:    {latest['timestamp']}")
                report.append("")
        
        return "\n".join(report)