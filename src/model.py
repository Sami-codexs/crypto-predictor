import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class CryptoLSTM:
    """
    LSTM neural network for crypto price direction prediction.
    """
    
    def __init__(self, sequence_length: int = 24, n_features: int = 12):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def build_model(self, lstm_units: int = 50, dropout_rate: float = 0.2):
        """Build LSTM architecture."""
        logger.info(f"Building LSTM: {lstm_units} units, {dropout_rate} dropout")
        
        model = Sequential([
            LSTM(units=lstm_units, return_sequences=False, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(dropout_rate),
            Dense(units=25, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Model built: {model.count_params()} parameters")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32) -> dict:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info(f"Training: {len(X_train)} samples, {epochs} epochs")
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        final_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history.get('val_accuracy', [0])[-1]
        
        logger.info(f"Training complete: acc={final_acc:.3f}, val_acc={final_val_acc:.3f}")
        
        return {
            'final_accuracy': final_acc,
            'final_val_accuracy': final_val_acc,
            'epochs_trained': len(self.history.history['loss'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (0-1)."""
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate on test set."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = self.predict(X_test)
        predicted_classes = (predictions > 0.5).astype(int)
        
        true_pos = np.sum((predicted_classes == 1) & (y_test == 1))
        false_pos = np.sum((predicted_classes == 1) & (y_test == 0))
        false_neg = np.sum((predicted_classes == 0) & (y_test == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save(self, filename: str = None) -> str:
        """Save model to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_lstm_{timestamp}.keras"
        
        filepath = os.path.join(self.model_dir, filename)
        self.model.save(filepath)
        
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'timestamp': datetime.now().isoformat(),
            'history': {k: [float(v) for v in vals] for k, vals in self.history.history.items()} if self.history else {}
        }
        
        meta_path = filepath.replace('.keras', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {filepath}")
        return filepath
    
    def load(self, filepath: str):
        """Load model from disk."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded: {filepath}")
        
        meta_path = filepath.replace('.keras', '_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                self.n_features = metadata.get('n_features', self.n_features)