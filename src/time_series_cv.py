"""
Day 17: Time Series Cross-Validation

Why this matters:
- Regular K-Fold shuffles data → leaks future information into training
- TimeSeriesSplit respects temporal order → realistic performance estimate
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """
    Walk-forward validation for time series models.
    Simulates real deployment: train on past, validate on future, repeat.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 24):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def create_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time-ordered train/validation splits using expanding window."""
        n_samples = len(df)
        min_required = self.n_splits * self.test_size + 50
        
        if n_samples < min_required:
            raise ValueError(f"Need {min_required} samples, have {n_samples}")
        
        splits = []
        for i in range(1, self.n_splits + 1):
            val_start = n_samples - (self.n_splits - i + 1) * self.test_size
            val_end = val_start + self.test_size
            
            train_df = df.iloc[:val_start].copy()
            val_df = df.iloc[val_start:val_end].copy()
            
            splits.append((train_df, val_df))
            logger.info(f"Split {i}: Train {len(train_df)}, Val {len(val_df)}")
        
        return splits
    
    def cross_validate(self, df, model_factory, preprocessor, coin_id="bitcoin") -> Dict:
        """Run walk-forward cross-validation."""
        logger.info("=" * 60)
        logger.info(f"Time Series Cross-Validation: {self.n_splits} splits")
        logger.info("=" * 60)
        
        splits = self.create_splits(df)
        fold_results = []
        
        for fold_idx, (train_df, val_df) in enumerate(splits, 1):
            logger.info(f"\n--- Fold {fold_idx}/{self.n_splits} ---")
            
            try:
                # Prepare sequences
                X_train, y_train = preprocessor.create_sequences(train_df)
                X_val, y_val = preprocessor.create_sequences(val_df)
                
                # Scale (fit on train only!)
                scaler = MinMaxScaler()
                n_samples, timesteps, features = X_train.shape
                X_train_scaled = scaler.fit_transform(
                    X_train.reshape(-1, features)
                ).reshape(n_samples, timesteps, features)
                
                n_val = X_val.shape[0]
                X_val_scaled = scaler.transform(
                    X_val.reshape(-1, features)
                ).reshape(n_val, timesteps, features)
                
                # Build fresh model
                model = model_factory()
                
                # Train
                model.train(X_train_scaled, y_train, X_val_scaled, y_val,
                          epochs=50, batch_size=4, use_early_stopping=True)
                
                # Evaluate
                y_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
                
                metrics = {
                    'fold': fold_idx,
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1': f1_score(y_val, y_pred, zero_division=0),
                }
                fold_results.append(metrics)
                logger.info(f"Fold {fold_idx}: Acc={metrics['accuracy']:.3f}")
                
            except Exception as e:
                logger.error(f"Fold {fold_idx} failed: {e}")
                continue
        
        return {
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_f1': np.mean([r['f1'] for r in fold_results]),
            'fold_results': fold_results
        }


def compare_validation_strategies(df, preprocessor, model_factory):
    """Compare TimeSeriesSplit vs random split to show the difference."""
    from sklearn.model_selection import train_test_split
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING VALIDATION STRATEGIES")
    logger.info("=" * 60)
    
    # Random split (WRONG)
    X, y = preprocessor.create_sequences(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    model = model_factory()
    model.train(X_train_s, y_train, epochs=30, batch_size=4, use_early_stopping=False)
    y_pred = (model.predict(X_test_s) > 0.5).astype(int)
    acc_rand = accuracy_score(y_test, y_pred)
    
    logger.info(f"Random Split Accuracy: {acc_rand:.3f} ⚠️")
    
    # Time series split (CORRECT)
    validator = TimeSeriesValidator(n_splits=3, test_size=24)
    ts_results = validator.cross_validate(df, model_factory, preprocessor)
    acc_ts = ts_results['mean_accuracy']
    
    logger.info(f"Time Series CV Accuracy: {acc_ts:.3f} ✅")
    logger.info(f"Optimism Bias: {abs(acc_rand - acc_ts):.3f} ({abs(acc_rand - acc_ts)/acc_ts*100:.1f}%)")
    
    return {
        'random_split_accuracy': acc_rand,
        'time_series_cv_accuracy': acc_ts,
        'difference': abs(acc_rand - acc_ts)
    }