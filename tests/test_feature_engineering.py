"""
Test suite for feature engineering pipeline.
Tests the FeatureEngine and TechnicalIndicators modules comprehensively.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.features.feature_engine import FeatureEngine
from src.features.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test all technical indicators implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        cumulative_returns = np.cumprod(1 + returns)
        close_prices = base_price * cumulative_returns

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": close_prices + np.random.normal(0, 100, 100),
                "high": close_prices + np.abs(np.random.normal(200, 100, 100)),
                "low": close_prices - np.abs(np.random.normal(200, 100, 100)),
                "close": close_prices,
                "volume": np.random.uniform(1000, 10000, 100),
            }
        )

        # Ensure high >= close >= low and high >= open >= low
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data.set_index("timestamp")

    def test_sma_calculation(self, sample_data):
        """Test Simple Moving Average calculation."""
        indicators = TechnicalIndicators()

        sma_20 = indicators.sma(sample_data["close"], 20)

        assert len(sma_20) == len(sample_data)
        assert not sma_20.isna().all()
        assert sma_20.iloc[19:].notna().all()  # Should have values after window period

        # Test that SMA is indeed the average
        manual_sma = sample_data["close"].iloc[0:20].mean()
        assert abs(sma_20.iloc[19] - manual_sma) < 1e-10

    def test_ema_calculation(self, sample_data):
        """Test Exponential Moving Average calculation."""
        indicators = TechnicalIndicators()

        ema_12 = indicators.ema(sample_data["close"], 12)

        assert len(ema_12) == len(sample_data)
        assert not ema_12.isna().any()
        assert ema_12.name == "EMA_12"

        # EMA should start with first value
        assert ema_12.iloc[0] == sample_data["close"].iloc[0]

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation."""
        indicators = TechnicalIndicators()

        rsi = indicators.rsi(sample_data["close"], 14)

        assert len(rsi) == len(sample_data)
        assert (rsi >= 0).all() and (rsi <= 100).all()
        assert not rsi.isna().any()  # Should fill NaN with neutral value

    def test_macd_calculation(self, sample_data):
        """Test MACD calculation."""
        indicators = TechnicalIndicators()

        macd_data = indicators.macd(sample_data["close"])

        assert isinstance(macd_data, pd.DataFrame)
        assert "MACD" in macd_data.columns
        assert "Signal" in macd_data.columns
        assert "Histogram" in macd_data.columns
        assert len(macd_data) == len(sample_data)

    def test_bollinger_bands_calculation(self, sample_data):
        """Test Bollinger Bands calculation."""
        indicators = TechnicalIndicators()

        bb_data = indicators.bollinger_bands(sample_data["close"])

        assert isinstance(bb_data, pd.DataFrame)
        required_cols = ["BB_Upper", "BB_Middle", "BB_Lower", "BB_Width", "BB_Position"]
        assert all(col in bb_data.columns for col in required_cols)

        # Upper should be >= Middle >= Lower (excluding NaN values)
        valid_data = bb_data.dropna()
        assert (valid_data["BB_Upper"] >= valid_data["BB_Middle"]).all()
        assert (valid_data["BB_Middle"] >= valid_data["BB_Lower"]).all()

    def test_atr_calculation(self, sample_data):
        """Test Average True Range calculation."""
        indicators = TechnicalIndicators()

        atr = indicators.atr(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )

        assert len(atr) == len(sample_data)
        # ATR should be non-negative (excluding NaN values)
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
        # Should have some non-NaN values
        assert len(valid_atr) > 0

    def test_stochastic_calculation(self, sample_data):
        """Test Stochastic Oscillator calculation."""
        indicators = TechnicalIndicators()

        stoch_data = indicators.stochastic(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )

        assert isinstance(stoch_data, pd.DataFrame)
        assert "Stoch_K" in stoch_data.columns
        assert "Stoch_D" in stoch_data.columns

        # Stochastic values should be between 0 and 100
        assert (stoch_data["Stoch_K"] >= 0).all() and (
            stoch_data["Stoch_K"] <= 100
        ).all()
        assert (stoch_data["Stoch_D"] >= 0).all() and (
            stoch_data["Stoch_D"] <= 100
        ).all()

    def test_williams_r_calculation(self, sample_data):
        """Test Williams %R calculation."""
        indicators = TechnicalIndicators()

        wr = indicators.williams_r(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )

        assert len(wr) == len(sample_data)
        assert (wr >= -100).all() and (wr <= 0).all()  # Williams %R range

    def test_obv_calculation(self, sample_data):
        """Test On Balance Volume calculation."""
        indicators = TechnicalIndicators()

        obv = indicators.obv(sample_data["close"], sample_data["volume"])

        assert len(obv) == len(sample_data)
        assert obv.name == "OBV"
        assert obv.iloc[0] == 0  # Should start at 0

    def test_adx_calculation(self, sample_data):
        """Test ADX calculation."""
        indicators = TechnicalIndicators()

        adx_data = indicators.adx(
            sample_data["high"], sample_data["low"], sample_data["close"]
        )

        assert isinstance(adx_data, pd.DataFrame)
        assert "ADX" in adx_data.columns
        assert "Plus_DI" in adx_data.columns
        assert "Minus_DI" in adx_data.columns

        # ADX and DI values should be non-negative (excluding NaN values)
        valid_data = adx_data.dropna()
        assert (valid_data["ADX"] >= 0).all()
        assert (valid_data["Plus_DI"] >= 0).all()
        assert (valid_data["Minus_DI"] >= 0).all()
        # Should have some valid data
        assert len(valid_data) > 0

    def test_vwap_calculation(self, sample_data):
        """Test VWAP calculation."""
        indicators = TechnicalIndicators()

        vwap = indicators.vwap(
            sample_data["high"],
            sample_data["low"],
            sample_data["close"],
            sample_data["volume"],
        )

        assert len(vwap) == len(sample_data)
        assert (vwap > 0).all()  # VWAP should be positive

    def test_calculate_all_indicators(self, sample_data):
        """Test that all indicators can be calculated together."""
        indicators = TechnicalIndicators()

        all_indicators = indicators.calculate_all_indicators(
            sample_data["high"],
            sample_data["low"],
            sample_data["close"],
            sample_data["volume"],
        )

        assert isinstance(all_indicators, pd.DataFrame)
        assert len(all_indicators) == len(sample_data)
        assert len(all_indicators.columns) > 20  # Should have many indicators

        # Check some key indicators exist
        key_indicators = ["SMA_20", "EMA_20", "RSI_14", "MACD", "BB_Upper", "ATR_14"]
        assert all(indicator in all_indicators.columns for indicator in key_indicators)


class TestFeatureEngine:
    """Test the FeatureEngine class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=200, freq="1h")
        np.random.seed(42)

        # Generate realistic price data
        base_price = 50000
        returns = np.random.normal(0, 0.02, 200)
        cumulative_returns = np.cumprod(1 + returns)
        close_prices = base_price * cumulative_returns

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": close_prices + np.random.normal(0, 100, 200),
                "high": close_prices + np.abs(np.random.normal(200, 100, 200)),
                "low": close_prices - np.abs(np.random.normal(200, 100, 200)),
                "close": close_prices,
                "volume": np.random.uniform(1000, 10000, 200),
            }
        )

        # Ensure price consistency
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data.set_index("timestamp")

    def test_feature_engine_initialization(self):
        """Test FeatureEngine initialization."""
        engine = FeatureEngine()

        assert engine.technical_indicators is not None
        assert engine.scaler is None
        assert engine.selected_features is None

    def test_create_price_features(self, sample_data):
        """Test price feature creation."""
        engine = FeatureEngine()

        price_features = engine.create_price_features(sample_data)

        assert isinstance(price_features, pd.DataFrame)
        assert len(price_features) == len(sample_data)

        # Check key price features exist
        expected_features = [
            "open",
            "high",
            "low",
            "close",
            "high_low_ratio",
            "close_open_ratio",
            "body_size",
            "price_change",
        ]
        assert all(feature in price_features.columns for feature in expected_features)

        # Check return features
        return_features = [col for col in price_features.columns if "return" in col]
        assert len(return_features) > 0

    def test_create_volume_features(self, sample_data):
        """Test volume feature creation."""
        engine = FeatureEngine()

        volume_features = engine.create_volume_features(sample_data)

        assert isinstance(volume_features, pd.DataFrame)
        assert len(volume_features) == len(sample_data)

        # Check key volume features exist
        expected_features = ["volume", "volume_change", "volume_price_trend"]
        assert all(feature in volume_features.columns for feature in expected_features)

        # Check volume ratio features
        ratio_features = [
            col for col in volume_features.columns if "volume_ratio" in col
        ]
        assert len(ratio_features) > 0

    def test_create_volatility_features(self, sample_data):
        """Test volatility feature creation."""
        engine = FeatureEngine()

        vol_features = engine.create_volatility_features(sample_data)

        assert isinstance(vol_features, pd.DataFrame)
        assert len(vol_features) == len(sample_data)

        # Check key volatility features exist
        expected_features = ["hist_vol_20d", "atr_normalized", "vol_ratio_short_long"]
        assert all(feature in vol_features.columns for feature in expected_features)

    def test_create_time_features(self, sample_data):
        """Test time feature creation."""
        engine = FeatureEngine()

        time_features = engine.create_time_features(sample_data)

        assert isinstance(time_features, pd.DataFrame)
        assert len(time_features) == len(sample_data)

        # Check time features exist
        expected_features = [
            "hour",
            "day_of_week",
            "month",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_asian_session",
            "is_european_session",
        ]
        assert all(feature in time_features.columns for feature in expected_features)

        # Check cyclical features are in correct range
        assert (time_features["hour_sin"] >= -1).all() and (
            time_features["hour_sin"] <= 1
        ).all()
        assert (time_features["hour_cos"] >= -1).all() and (
            time_features["hour_cos"] <= 1
        ).all()

    def test_create_lag_features(self, sample_data):
        """Test lag feature creation."""
        engine = FeatureEngine()

        # First create basic features
        price_features = engine.create_price_features(sample_data)

        lag_features = engine.create_lag_features(price_features)

        assert isinstance(lag_features, pd.DataFrame)
        assert len(lag_features) == len(sample_data)

        # Check lag features exist
        lag_cols = [col for col in lag_features.columns if "_lag_" in col]
        assert len(lag_cols) > 0

    def test_create_all_features(self, sample_data):
        """Test comprehensive feature creation."""
        engine = FeatureEngine()

        all_features = engine.create_all_features(sample_data)

        assert isinstance(all_features, pd.DataFrame)
        assert len(all_features) == len(sample_data)
        assert len(all_features.columns) >= 50  # Should create many features

        # Check no infinite or excessive NaN values
        assert not np.isinf(all_features.select_dtypes(include=[np.number])).any().any()
        nan_percentage = all_features.isna().sum().sum() / (
            len(all_features) * len(all_features.columns)
        )
        assert nan_percentage < 0.1  # Less than 10% NaN values

    def test_feature_selection(self, sample_data):
        """Test feature selection functionality."""
        engine = FeatureEngine()

        # Create features
        features = engine.create_all_features(sample_data)

        # Create dummy target (binary classification)
        np.random.seed(42)
        target = pd.Series(
            np.random.binomial(1, 0.5, len(features)), index=features.index
        )

        # Test feature selection
        selected_features = engine.select_features(
            features, target, method="mutual_info", top_k=20
        )

        assert isinstance(selected_features, list)
        assert len(selected_features) == 20
        assert all(feature in features.columns for feature in selected_features)

    def test_scaler_functionality(self, sample_data):
        """Test scaler fitting and transformation."""
        engine = FeatureEngine()

        # Create features
        features = engine.create_all_features(sample_data)

        # Fit scaler
        engine.fit_scaler(features, method="robust")
        assert engine.scaler is not None

        # Transform features
        transformed = engine.transform_features(features)
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(features)

    def test_memory_usage_report(self, sample_data):
        """Test memory usage reporting."""
        engine = FeatureEngine()

        features = engine.create_all_features(sample_data)
        memory_report = engine.memory_usage_report(features)

        assert isinstance(memory_report, dict)
        required_keys = ["total_memory_mb", "num_features", "num_samples"]
        assert all(key in memory_report for key in required_keys)
        assert memory_report["num_features"] == len(features.columns)
        assert memory_report["num_samples"] == len(features)


class TestIntegration:
    """Integration tests for the feature engineering pipeline."""

    @pytest.fixture
    def realistic_data(self):
        """Create realistic crypto market data."""
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="1h")
        np.random.seed(42)

        # Simulate more realistic crypto price movements
        base_price = 50000
        trend = np.linspace(0, 0.5, 1000)  # Upward trend
        noise = np.random.normal(0, 0.03, 1000)
        cyclical = 0.1 * np.sin(
            np.linspace(0, 20 * np.pi, 1000)
        )  # Some cyclical patterns

        price_changes = trend + noise + cyclical
        prices = base_price * np.cumprod(1 + price_changes)

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices + np.random.normal(0, 50, 1000),
                "high": prices + np.abs(np.random.normal(100, 50, 1000)),
                "low": prices - np.abs(np.random.normal(100, 50, 1000)),
                "close": prices,
                "volume": np.random.lognormal(
                    8, 1, 1000
                ),  # Log-normal volume distribution
            }
        )

        # Ensure price consistency
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data.set_index("timestamp")

    def test_end_to_end_pipeline(self, realistic_data):
        """Test the complete feature engineering pipeline."""
        engine = FeatureEngine()

        # Step 1: Create all features
        features = engine.create_all_features(realistic_data)

        # Step 2: Create target variable (simplified price direction)
        future_returns = (
            realistic_data["close"].pct_change(4).shift(-4)
        )  # 4-hour forward return
        target = (future_returns > 0).astype(int)
        target = target.dropna()

        # Align features and target
        min_length = min(len(features), len(target))
        features_aligned = features.iloc[:min_length]
        target_aligned = target.iloc[:min_length]

        # Step 3: Feature selection
        selected_features = engine.select_features(
            features_aligned, target_aligned, method="mutual_info", top_k=30
        )

        # Step 4: Fit scaler
        engine.fit_scaler(features_aligned[selected_features], method="robust")

        # Step 5: Transform features
        final_features = engine.transform_features(features_aligned[selected_features])

        # Assertions
        assert len(final_features) == len(target_aligned)
        assert len(final_features.columns) == 30
        assert not final_features.isna().any().any()
        assert not np.isinf(final_features).any().any()

        # Check that features have reasonable scale after transformation
        feature_stats = final_features.describe()
        assert (
            feature_stats.loc["std"] > 0
        ).all()  # All features should have variation


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
