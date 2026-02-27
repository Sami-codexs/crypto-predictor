import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.model_selection import ParameterGrid
from src.model import CryptoLSTM
from src.preprocessing import MLPreprocessor
import tensorflow as tf

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Hyperparameter optimization for LSTM.
    """
    
    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.best_score = 0
    
    def grid_search(self, X_train, y_train, X_val, y_val, 
                   param_grid: Dict = None) -> Tuple[Dict, float]:
        """
        Search best hyperparameters.
        """
        if param_grid is None:
            param_grid = {
                'lstm_units': [30, 50, 70],
                'dropout_rate': [0.2, 0.3, 0.5],
                'learning_rate': [0.001, 0.0005],
                'batch_size': [4, 8]
            }
        
        grid = ParameterGrid(param_grid)
        results = []
        
        logger.info(f"Grid search: {len(grid)} combinations")
        
        for i, params in enumerate(grid, 1):
            logger.info(f"\nTesting {i}/{len(grid)}: {params}")
            
            # Build and train
            model = CryptoLSTM(sequence_length=24, n_features=X_train.shape[2])
            model.build_model(
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate']
            )
            
            # Update learning rate
            model.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train
            history = model.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=params['batch_size'],
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            results.append((params, val_acc))
            
            logger.info(f"  Val accuracy: {val_acc:.3f}")
            
            if val_acc > self.best_score:
                self.best_score = val_acc
                self.best_params = params
                self.best_model = model
        
        logger.info(f"\nBest: {self.best_params} with {self.best_score:.3f} val accuracy")
        return self.best_params, self.best_score
    
    def train_with_class_weights(self, X_train, y_train, X_val, y_val, 
                                  class_weight: Dict = None) -> CryptoLSTM:
        """
        Train with class weights to fix imbalance.
        """
        if class_weight is None:
            # Auto-calculate
            from sklearn.utils.class_weight import compute_class_weight
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight = {i: w for i, w in enumerate(weights)}
        
        logger.info(f"Using class weights: {class_weight}")
        
        model = CryptoLSTM(sequence_length=24, n_features=X_train.shape[2])
        model.build_model(lstm_units=50, dropout_rate=0.3)
        
        history = model.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=8,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=1
        )
        
        return model