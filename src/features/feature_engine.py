"""
Feature engineering pipeline for cryptocurrency trading.
Creates comprehensive features from OHLCV data and technical indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

from .technical_indicators import TechnicalIndicators
from ..utils.logging import get_logger


class FeatureEngine:
    """
    Comprehensive feature engineering pipeline.
    Creates features from price data, volume, and technical indicators.
    """
    
    def __init__(self):
        """Initialize feature engineering pipeline."""
        self.logger = get_logger().get_logger()
        self.technical_indicators = TechnicalIndicators()
        self.scaler = None
        self.selected_features = None
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['open'] = data['open']
        features['high'] = data['high'] 
        features['low'] = data['low']
        features['close'] = data['close']
        
        # Price relationships
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        features['high_close_ratio'] = data['high'] / data['close']
        features['low_close_ratio'] = data['low'] / data['close']
        
        # Intraday measures
        features['body_size'] = np.abs(data['close'] - data['open']) / data['open']
        features['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['open']
        features['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['open']
        features['total_range'] = (data['high'] - data['low']) / data['open']
        
        # Price changes
        features['price_change'] = data['close'].pct_change()
        features['price_change_abs'] = np.abs(features['price_change'])
        
        # Multi-period returns
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'return_{period}d'] = data['close'].pct_change(periods=period)
            features[f'log_return_{period}d'] = np.log(data['close'] / data['close'].shift(period))
            
        # Volatility measures
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}d'] = data['close'].pct_change().rolling(window).std()
            features[f'price_std_{window}d'] = data['close'].rolling(window).std() / data['close'].rolling(window).mean()
        
        self.logger.info(f"Created {len(features.columns)} price features")
        return features
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic volume features
        features['volume'] = data['volume']
        features['volume_change'] = data['volume'].pct_change()
        
        # Volume ratios and normalizations
        for window in [5, 10, 20, 50]:
            vol_mean = data['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}d'] = data['volume'] / vol_mean
            features[f'volume_std_{window}d'] = data['volume'].rolling(window).std() / vol_mean
        
        # Price-volume relationships
        features['volume_price_trend'] = (data['close'].pct_change() * data['volume']).rolling(5).mean()
        features['price_volume_correlation'] = data['close'].rolling(20).corr(data['volume'])
        
        # Volume momentum
        for period in [5, 10, 20]:
            features[f'volume_momentum_{period}d'] = data['volume'] / data['volume'].shift(period)
        
        # Volume-weighted prices
        features['vwap_deviation'] = (data['close'] - self.technical_indicators.vwap(
            data['high'], data['low'], data['close'], data['volume']
        )) / data['close']
        
        self.logger.info(f"Created {len(features.columns)} volume features")
        return features
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        
        # Historical volatility (different windows)
        for window in [5, 10, 20, 50, 100]:
            returns = data['close'].pct_change()
            features[f'hist_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)  # Annualized
            
        # ATR-based volatility
        atr = self.technical_indicators.atr(data['high'], data['low'], data['close'])
        features['atr_normalized'] = atr / data['close']
        
        # Volatility ratios
        features['vol_ratio_short_long'] = features['hist_vol_5d'] / features['hist_vol_50d']
        features['vol_ratio_mid_long'] = features['hist_vol_20d'] / features['hist_vol_50d']
        
        # Volatility percentiles
        for window in [50, 100, 200]:
            vol_20 = features['hist_vol_20d']
            features[f'vol_percentile_{window}d'] = vol_20.rolling(window).rank(pct=True)
        
        # GARCH-like features
        returns = data['close'].pct_change().fillna(0)
        features['squared_returns'] = returns ** 2
        features['garch_vol'] = features['squared_returns'].ewm(alpha=0.1).mean()
        
        self.logger.info(f"Created {len(features.columns)} volatility features")
        return features
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            data: DataFrame with features
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        lag_features = pd.DataFrame(index=data.index)
        
        # Select key columns for lagging
        key_columns = ['close', 'volume', 'price_change', 'volatility_20d', 'volume_ratio_20d']
        
        for col in key_columns:
            if col in data.columns:
                for lag in lags:
                    lag_features[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        self.logger.info(f"Created {len(lag_features.columns)} lag features")
        return lag_features
    
    def create_rolling_statistics(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            data: DataFrame with base features
            windows: List of rolling windows
            
        Returns:
            DataFrame with rolling statistics
        """
        rolling_features = pd.DataFrame(index=data.index)
        
        # Key columns for rolling statistics
        key_columns = ['close', 'volume']
        
        for col in key_columns:
            if col in data.columns:
                for window in windows:
                    rolling_features[f'{col}_mean_{window}'] = data[col].rolling(window).mean()
                    rolling_features[f'{col}_std_{window}'] = data[col].rolling(window).std()
                    rolling_features[f'{col}_min_{window}'] = data[col].rolling(window).min()
                    rolling_features[f'{col}_max_{window}'] = data[col].rolling(window).max()
                    rolling_features[f'{col}_median_{window}'] = data[col].rolling(window).median()
                    rolling_features[f'{col}_skew_{window}'] = data[col].rolling(window).skew()
                    rolling_features[f'{col}_kurt_{window}'] = data[col].rolling(window).kurt()
                    
                    # Percentile features
                    rolling_features[f'{col}_pct25_{window}'] = data[col].rolling(window).quantile(0.25)
                    rolling_features[f'{col}_pct75_{window}'] = data[col].rolling(window).quantile(0.75)
                    
                    # Position in range
                    rolling_min = data[col].rolling(window).min()
                    rolling_max = data[col].rolling(window).max()
                    rolling_features[f'{col}_position_{window}'] = (data[col] - rolling_min) / (rolling_max - rolling_min)
        
        self.logger.info(f"Created {len(rolling_features.columns)} rolling statistics features")
        return rolling_features
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            data: DataFrame with timestamp index
            
        Returns:
            DataFrame with time features
        """
        features = pd.DataFrame(index=data.index)
        
        # Extract time components
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['day_of_month'] = data.index.day
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        
        # Cyclical encoding for time features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Market session indicators (assuming UTC timestamps)
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)  # Crypto trades 24/7 but this can be useful
        features['is_asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['is_european_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['is_american_session'] = ((features['hour'] >= 16) & (features['hour'] < 24)).astype(int)
        
        self.logger.info(f"Created {len(features.columns)} time features")
        return features
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features using the complete pipeline.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("Starting comprehensive feature engineering...")
        
        # Validate input data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Initialize features DataFrame
        all_features = pd.DataFrame(index=data.index)
        
        # Create different feature groups
        price_features = self.create_price_features(data)
        volume_features = self.create_volume_features(data)
        volatility_features = self.create_volatility_features(data)
        time_features = self.create_time_features(data)
        
        # Technical indicators
        technical_features = self.technical_indicators.calculate_all_indicators(
            data['high'], data['low'], data['close'], data['volume']
        )
        
        # Combine all feature groups
        feature_groups = [
            price_features,
            volume_features, 
            volatility_features,
            time_features,
            technical_features
        ]
        
        for features in feature_groups:
            all_features = pd.concat([all_features, features], axis=1)
        
        # Create lag features from the combined features
        key_features_for_lags = all_features[['close', 'volume', 'price_change', 'RSI_14', 'MACD']].copy()
        lag_features = self.create_lag_features(key_features_for_lags)
        all_features = pd.concat([all_features, lag_features], axis=1)
        
        # Create rolling statistics for key features
        rolling_stats = self.create_rolling_statistics(data)
        all_features = pd.concat([all_features, rolling_stats], axis=1)
        
        # Remove any duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Replace infinite values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values, then backward fill, then fill remaining with 0
        all_features = all_features.ffill().bfill().fillna(0)
        
        self.logger.info(f"Feature engineering completed. Created {len(all_features.columns)} total features")
        return all_features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', top_k: int = 50) -> List[str]:
        """
        Select most important features.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('mutual_info', 'f_score', 'correlation')
            top_k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting top {top_k} features using {method}")
        
        # Remove any remaining NaN or infinite values
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_clean = y.fillna(0)
        
        if method == 'mutual_info':
            # Use mutual information for feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(top_k, X_clean.shape[1]))
            selector.fit(X_clean, y_clean)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
        elif method == 'f_score':
            # Use F-score for feature selection
            selector = SelectKBest(score_func=f_classif, k=min(top_k, X_clean.shape[1]))
            selector.fit(X_clean, y_clean)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
        elif method == 'correlation':
            # Use correlation with target
            correlations = X_clean.corrwith(y_clean).abs().sort_values(ascending=False)
            selected_features = correlations.head(top_k).index.tolist()
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.selected_features = selected_features
        self.logger.info(f"Selected {len(selected_features)} features")
        return selected_features
    
    def fit_scaler(self, X: pd.DataFrame, method: str = 'robust') -> None:
        """
        Fit data scaler.
        
        Args:
            X: Feature matrix to fit scaler on
            method: Scaling method ('standard', 'robust')
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit scaler on clean data
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.scaler.fit(X_clean)
        self.logger.info(f"Fitted {method} scaler")
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler and selected features.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Scaled and selected features
        """
        # Clean data
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Select features if available
        if self.selected_features:
            available_features = [f for f in self.selected_features if f in X_clean.columns]
            X_selected = X_clean[available_features]
        else:
            X_selected = X_clean
        
        # Scale features if scaler is fitted
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_selected),
                index=X_selected.index,
                columns=X_selected.columns
            )
        else:
            X_scaled = X_selected
        
        return X_scaled
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
    
    def memory_usage_report(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate memory usage report for features.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Dictionary with memory usage statistics
        """
        memory_usage = df.memory_usage(deep=True)
        
        report = {
            'total_memory_mb': memory_usage.sum() / (1024**2),
            'mean_memory_per_column_kb': memory_usage.mean() / 1024,
            'max_memory_column_mb': memory_usage.max() / (1024**2),
            'num_features': len(df.columns),
            'num_samples': len(df)
        }
        
        self.logger.info(f"Memory usage: {report['total_memory_mb']:.2f} MB for {report['num_features']} features")
        return report