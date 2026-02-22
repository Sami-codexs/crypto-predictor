import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class MLPreprocessor:
    """
    Prepares time-series data for LSTM training.
    Critical: No lookahead bias, proper scaling, sequence creation.
    """
    def create_overlapping_sequences(self, df: pd.DataFrame, target_col: str = 'target', 
                                  stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with overlapping windows.
        
        Normal: [0-24], [24-48] = 2 sequences from 48 rows
        Overlap stride=1: [0-24], [1-25], [2-26]... = 24 sequences from 48 rows
        """
        feature_cols = [col for col in df.columns 
                    if col not in ['timestamp', target_col, 'coin_id']]
        self.feature_columns = feature_cols
        
        data = df[feature_cols].values
        targets = df[target_col].values
        
        X, y = [], []
        
        # Create overlapping windows with stride
        for i in range(0, len(data) - self.sequence_length, stride):
            X.append(data[i:(i + self.sequence_length)])
            y.append(targets[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} overlapping sequences (stride={stride})")
        return X, y
    
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length  # 24 hours of history
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
    
    def create_sequences(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM.
        
        X shape: (num_samples, sequence_length, num_features)
        y shape: (num_samples,)
        
        Each X[i] is 24h of data, y[i] is what happens next hour.
        """
        # Select feature columns (exclude timestamp and target)
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', target_col, 'coin_id']]
        self.feature_columns = feature_cols
        
        data = df[feature_cols].values
        targets = df[target_col].values
        
        X, y = [], []
        
        # Create sliding windows
        for i in range(len(data) - self.sequence_length):
            # X: sequence_length rows of features
            X.append(data[i:(i + self.sequence_length)])
            # y: target immediately after sequence
            y.append(targets[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences of length {self.sequence_length}")
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features to 0-1 range.
        Fit on training data only, transform both.
        """
        # Reshape for scaling: (samples * timesteps, features)
        num_samples, timesteps, features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, features)
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(num_samples, timesteps, features)
        
        if X_test is not None:
            # Transform test data with same scaler
            num_test_samples = X_test.shape[0]
            X_test_reshaped = X_test.reshape(-1, features)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(num_test_samples, timesteps, features)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def split_train_test(self, X: np.ndarray, y: np.ndarray, 
                         test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Time-ordered train/test split.
        NEVER shuffle time-series data!
        """
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(self, df: pd.DataFrame, use_overlap: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full pipeline with optional overlapping sequences.
        """
        df_clean = df.dropna()
        logger.info(f"Clean data: {len(df_clean)} rows")
        
        # Use overlapping sequences if data is scarce
        if use_overlap and len(df_clean) < 100:
            X, y = self.create_overlapping_sequences(df_clean, stride=2)
            logger.info("Using overlapping sequences (data augmentation)")
        else:
            X, y = self.create_sequences(df_clean)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        X_train, X_test, y_train, y_test = self.split_train_test(X, y, test_size=0.2)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def inverse_scale_prediction(self, scaled_value: float) -> float:
        """Convert scaled prediction back to original scale if needed."""
        # For classification (0/1), scaling doesn't change much
        # But kept for consistency
        return scaled_value