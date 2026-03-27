#!/usr/bin/env python3
"""
Day 19: Testing Suite with pytest

Production-grade testing for ML systems.
Works when placed in project root directory.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path (critical for imports to work)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import after path setup
from indicators import TechnicalIndicators
from preprocessing import MLPreprocessor
from model import CryptoLSTM
from validation import DataValidator
from database import CryptoDatabase


# =============================================================================
# FIXTURES (Defined directly in this file for root-directory execution)
# =============================================================================

@pytest.fixture(scope="session")
def validator():
    """Create DataValidator fixture."""
    return DataValidator()


@pytest.fixture(scope="session")
def sample_dataframe():
    """Create sample dataframe for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'price_usd': np.random.randn(100).cumsum() + 50000,
        'volume_24h': np.random.randint(1000000, 5000000, 100),
        'market_cap': np.random.randint(900000000, 1100000000, 100),
    })
    return df


@pytest.fixture(scope="session")
def preprocessor():
    """Create MLPreprocessor fixture."""
    return MLPreprocessor(sequence_length=24)


@pytest.fixture(scope="function")
def temp_database(tmp_path):
    """Create temporary database for testing."""
    db_path = str(tmp_path / "test_crypto.db")
    db = CryptoDatabase(db_path=db_path)
    yield db


# =============================================================================
# UNIT TESTS: Data Validation
# =============================================================================

class TestDataValidation:
    """Unit tests for data validation."""
    
    def test_validate_price_normal(self, validator):
        """Test normal price validation."""
        is_valid, msg = validator.validate_price(50000, None)
        assert is_valid is True
    
    def test_validate_price_none(self, validator):
        """Test None price validation."""
        is_valid, msg = validator.validate_price(None, None)
        assert is_valid is False
    
    def test_validate_price_negative(self, validator):
        """Test negative price validation."""
        is_valid, msg = validator.validate_price(-100, None)
        assert is_valid is False
    
    def test_validate_price_suspicious_jump(self, validator):
        """Test suspicious price jump validation."""
        is_valid, msg = validator.validate_price(50000, 40000)  # 25% jump
        assert is_valid is False
    
    def test_validate_price_normal_change(self, validator):
        """Test normal price change validation."""
        is_valid, msg = validator.validate_price(50000, 49000)  # 2% change
        assert is_valid is True
    
    def test_validate_dataframe_empty(self, validator):
        """Test validation of empty dataframe."""
        df = pd.DataFrame()
        is_valid, cleaned, errors = validator.validate_dataframe(df)
        assert not is_valid
        assert len(errors) > 0


# =============================================================================
# UNIT TESTS: Preprocessing
# =============================================================================

class TestPreprocessing:
    """Unit tests for data preprocessing."""
    
    def test_create_sequences_shape(self, sample_dataframe, preprocessor):
        """Test sequence creation produces correct shapes."""
        df = sample_dataframe.copy()
        df['target'] = (df['price_usd'].shift(-1) > df['price_usd']).astype(int)
        
        X, y = preprocessor.create_sequences(df)
        
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 24, f"Sequence length should be 24, got {X.shape[1]}"
        assert len(y.shape) == 1, "y should be 1D"
    
    def test_train_test_split_temporal(self, sample_dataframe, preprocessor):
        """Test that train/test split preserves temporal order."""
        df = sample_dataframe.copy()
        df['target'] = (df['price_usd'].shift(-1) > df['price_usd']).astype(int)
        
        X, y = preprocessor.create_sequences(df)
        X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y, test_size=0.2)
        
        assert len(X_train) > len(X_test), "Train set should be larger"
        assert len(X_train) + len(X_test) == len(X), "All data should be used"


# =============================================================================
# UNIT TESTS: Technical Indicators
# =============================================================================

class TestIndicators:
    """Unit tests for technical indicators."""
    
    def test_rsi_calculation_range(self, sample_dataframe):
        """Test RSI is always between 0 and 100."""
        ind = TechnicalIndicators()
        prices = sample_dataframe['price_usd']
        rsi = ind.calculate_rsi(prices)
        
        # Skip NaN values (warmup period)
        valid_rsi = rsi.dropna()
        assert len(valid_rsi) > 0, "RSI should have some valid values"
        assert valid_rsi.min() >= 0, f"RSI minimum should be >= 0, got {valid_rsi.min()}"
        assert valid_rsi.max() <= 100, f"RSI maximum should be <= 100, got {valid_rsi.max()}"
    
    def test_macd_calculation_components(self, sample_dataframe):
        """Test MACD calculation produces expected components."""
        ind = TechnicalIndicators()
        prices = sample_dataframe['price_usd']
        macd = ind.calculate_macd(prices)
        
        assert 'macd_line' in macd
        assert 'signal_line' in macd
        assert 'histogram' in macd
        assert len(macd['macd_line']) == len(prices)
    
    def test_bollinger_bands_structure(self, sample_dataframe):
        """Test Bollinger Bands calculation."""
        ind = TechnicalIndicators()
        prices = sample_dataframe['price_usd']
        bb = ind.calculate_bollinger_bands(prices)
        
        assert 'upper_band' in bb
        assert 'lower_band' in bb
        assert 'middle_band' in bb
        
        # Upper band should be >= lower band (where not NaN)
        valid_idx = ~(np.isnan(bb['upper_band']) | np.isnan(bb['lower_band']))
        if valid_idx.any():
            assert all(bb['upper_band'][valid_idx] >= bb['lower_band'][valid_idx])


# =============================================================================
# UNIT TESTS: Model
# =============================================================================

class TestModel:
    """Unit tests for LSTM model."""
    
    def test_model_build_architecture(self):
        """Test model can be built with correct architecture."""
        model = CryptoLSTM(sequence_length=24, n_features=12)
        model.build_model(lstm_units=50, dropout_rate=0.2)
        
        assert model.model is not None
        assert model.model.input_shape == (None, 24, 12)
        assert model.model.output_shape == (None, 1)
    
    def test_model_prediction_shape_and_range(self):
        """Test model produces correct prediction shape and valid probabilities."""
        model = CryptoLSTM(sequence_length=24, n_features=12)
        model.build_model(lstm_units=50, dropout_rate=0.2)
        
        # Create dummy input
        X = np.random.randn(5, 24, 12).astype(np.float32)
        predictions = model.predict(X)
        
        assert predictions.shape == (5,), f"Expected shape (5,), got {predictions.shape}"
        assert all((predictions >= 0) & (predictions <= 1)), "Predictions should be probabilities"
        assert not np.isnan(predictions).any(), "Predictions should not contain NaN"
    
    def test_model_save_and_load(self, tmp_path):
        """Test model can be saved and loaded."""
        model = CryptoLSTM(sequence_length=24, n_features=12)
        model.build_model(lstm_units=50, dropout_rate=0.2)
        
        # Save
        model_path = str(tmp_path / "test_model.keras")
        saved_path = model.save(model_path)
        
        assert os.path.exists(saved_path), "Model file should exist"
        
        # Load
        new_model = CryptoLSTM(sequence_length=24, n_features=12)
        new_model.load(saved_path)
        
        assert new_model.model is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_database_insert_and_retrieve(self, temp_database):
        """Test database insert and retrieve operations."""
        db = temp_database
        
        # Insert
        success = db.insert_price("bitcoin", price=50000.0, volume=1000000.0, market_cap=900000000000.0)
        assert success, "Insert should succeed"
        
        # Retrieve
        df = db.get_recent_prices("bitcoin", hours=24)
        assert len(df) >= 1, "Should retrieve at least 1 record"
        assert 'price_usd' in df.columns


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Run with: pytest test_day19.py -v
    pytest.main([__file__, "-v"])