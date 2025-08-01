"""
Market Regime Detection for Adaptive Parameter Optimization
=========================================================

Detects different market regimes (trending, ranging, volatile, calm) to enable
adaptive parameter selection. Different market conditions require different
technical indicator parameters for optimal performance.

Features:
- Multi-factor regime detection (volatility, trend strength, momentum)
- Hidden Markov Model for regime classification
- Real-time regime detection
- Regime transition analysis
- Parameter adaptation recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

from ..technical_indicators import TechnicalIndicators
from ...utils.logging import get_logger


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


@dataclass
class RegimeCharacteristics:
    """Characteristics of a market regime"""
    regime: MarketRegime
    volatility_percentile: float
    trend_strength: float
    momentum_strength: float
    mean_reversion_tendency: float
    regime_persistence: float
    recommended_indicators: List[str]


@dataclass
class RegimeDetectionResult:
    """Results of regime detection analysis"""
    regimes: pd.Series
    regime_probabilities: pd.DataFrame
    regime_characteristics: Dict[MarketRegime, RegimeCharacteristics]
    transition_matrix: np.ndarray
    regime_stability: float
    current_regime: MarketRegime


class MarketRegimeDetector:
    """
    Advanced market regime detector using multiple statistical approaches.
    
    Combines volatility analysis, trend detection, and momentum measurements
    to classify market conditions and recommend optimal indicator parameters.
    """
    
    def __init__(self, 
                 lookback_window: int = 50,
                 regime_memory: int = 10,
                 volatility_window: int = 20):
        """
        Initialize market regime detector.
        
        Args:
            lookback_window: Window for regime feature calculation
            regime_memory: Memory for regime smoothing
            volatility_window: Window for volatility calculation
        """
        self.lookback_window = lookback_window
        self.regime_memory = regime_memory
        self.volatility_window = volatility_window
        
        self.logger = get_logger().get_logger()
        self.technical_indicators = TechnicalIndicators()
        self.scaler = StandardScaler()
        
        # Regime detection models
        self.hmm_model = None
        self.gmm_model = None
        self.fitted = False
    
    def detect_regimes(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """
        Detect market regimes in historical data.
        
        Args:
            data: Historical OHLCV data
            
        Returns:
            RegimeDetectionResult with regime classification
        """
        self.logger.info("Starting market regime detection analysis")
        
        # Calculate regime features
        features = self._calculate_regime_features(data)
        
        # Detect regimes using multiple methods
        regime_labels = self._detect_regimes_hmm(features)
        regime_probabilities = self._calculate_regime_probabilities(features)
        
        # Convert numeric labels to regime enums
        regimes = self._map_labels_to_regimes(regime_labels, features)
        
        # Analyze regime characteristics
        regime_characteristics = self._analyze_regime_characteristics(data, regimes, features)
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regimes)
        
        # Calculate regime stability
        regime_stability = self._calculate_regime_stability(regimes)
        
        # Get current regime
        current_regime = regimes.iloc[-1] if len(regimes) > 0 else MarketRegime.RANGING
        
        result = RegimeDetectionResult(
            regimes=regimes,
            regime_probabilities=regime_probabilities,
            regime_characteristics=regime_characteristics,
            transition_matrix=transition_matrix,
            regime_stability=regime_stability,
            current_regime=current_regime
        )
        
        self.logger.info(f"Regime detection completed. Current regime: {current_regime.value}")
        return result
    
    def get_regime_specific_parameters(self, 
                                     regime: MarketRegime) -> Dict[str, Any]:
        """
        Get recommended parameters for a specific market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Dictionary of recommended parameter values
        """
        # Parameter recommendations based on regime characteristics
        regime_params = {
            MarketRegime.TRENDING_UP: {
                'rsi_period': 9,
                'rsi_oversold': 25,
                'rsi_overbought': 70,
                'macd_fast': 8,
                'macd_slow': 21,
                'macd_signal': 6,
                'bb_period': 15,
                'bb_std': 1.8,
                'ma_fast': 8,
                'ma_slow': 21,
                'atr_multiplier': 2.0,
                'strategy_type': 'momentum'
            },
            
            MarketRegime.TRENDING_DOWN: {
                'rsi_period': 9,
                'rsi_oversold': 30,
                'rsi_overbought': 75,
                'macd_fast': 8,
                'macd_slow': 21,
                'macd_signal': 6,
                'bb_period': 15,
                'bb_std': 1.8,
                'ma_fast': 8,
                'ma_slow': 21,
                'atr_multiplier': 2.5,
                'strategy_type': 'momentum'
            },
            
            MarketRegime.RANGING: {
                'rsi_period': 21,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 25,
                'bb_std': 2.2,
                'ma_fast': 20,
                'ma_slow': 50,
                'atr_multiplier': 1.5,
                'strategy_type': 'mean_reversion'
            },
            
            MarketRegime.HIGH_VOLATILITY: {
                'rsi_period': 14,
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'macd_fast': 8,
                'macd_slow': 17,
                'macd_signal': 6,
                'bb_period': 12,
                'bb_std': 2.5,
                'ma_fast': 5,
                'ma_slow': 15,
                'atr_multiplier': 3.0,
                'strategy_type': 'volatility_breakout'
            },
            
            MarketRegime.LOW_VOLATILITY: {
                'rsi_period': 30,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'macd_fast': 16,
                'macd_slow': 35,
                'macd_signal': 12,
                'bb_period': 30,
                'bb_std': 1.8,
                'ma_fast': 25,
                'ma_slow': 75,
                'atr_multiplier': 1.2,
                'strategy_type': 'mean_reversion'
            },
            
            MarketRegime.BREAKOUT: {
                'rsi_period': 7,
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'macd_fast': 6,
                'macd_slow': 15,
                'macd_signal': 4,
                'bb_period': 10,
                'bb_std': 2.0,
                'ma_fast': 5,
                'ma_slow': 12,
                'atr_multiplier': 2.8,
                'strategy_type': 'breakout'
            },
            
            MarketRegime.CONSOLIDATION: {
                'rsi_period': 25,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'macd_fast': 14,
                'macd_slow': 30,
                'macd_signal': 10,
                'bb_period': 28,
                'bb_std': 2.0,
                'ma_fast': 20,
                'ma_slow': 60,
                'atr_multiplier': 1.5,
                'strategy_type': 'consolidation'
            }
        }
        
        return regime_params.get(regime, regime_params[MarketRegime.RANGING])
    
    def predict_regime_transitions(self, 
                                 current_data: pd.DataFrame,
                                 forecast_periods: int = 5) -> List[Tuple[MarketRegime, float]]:
        """
        Predict likely regime transitions.
        
        Args:
            current_data: Recent market data
            forecast_periods: Number of periods to forecast
            
        Returns:
            List of (regime, probability) tuples for forecast periods
        """
        if not self.fitted or self.hmm_model is None:
            self.logger.warning("Model not fitted. Cannot predict regime transitions.")
            return []
        
        try:
            # Calculate features for current data
            features = self._calculate_regime_features(current_data)
            
            if len(features) == 0:
                return []
            
            # Use HMM to predict next states
            current_features = features.iloc[-min(10, len(features)):].values
            
            # Predict next regime states
            predictions = []
            for _ in range(forecast_periods):
                # This is a simplified prediction - in practice, you'd use HMM's predict method
                current_regime_probs = self.hmm_model.predict_proba(current_features)[-1]
                predicted_regime_idx = np.argmax(current_regime_probs)
                predicted_prob = current_regime_probs[predicted_regime_idx]
                
                # Map to regime enum
                predicted_regime = list(MarketRegime)[predicted_regime_idx % len(MarketRegime)]
                predictions.append((predicted_regime, predicted_prob))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting regime transitions: {e}")
            return []
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection"""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Price-based features
            returns = data['close'].pct_change()
            
            # Volatility features
            features['volatility'] = returns.rolling(self.volatility_window).std()
            features['volatility_percentile'] = features['volatility'].rolling(100).rank(pct=True)
            
            # Trend strength features
            sma_20 = self.technical_indicators.sma(data['close'], 20)
            sma_50 = self.technical_indicators.sma(data['close'], 50)
            features['trend_strength'] = (sma_20 - sma_50) / sma_50
            features['price_vs_sma20'] = (data['close'] - sma_20) / sma_20
            
            # Momentum features
            rsi = self.technical_indicators.rsi(data['close'], 14)
            features['rsi'] = rsi
            features['rsi_momentum'] = rsi.diff(5)
            
            # MACD features
            macd_data = self.technical_indicators.macd(data['close'])
            features['macd_signal'] = macd_data['MACD'] - macd_data['Signal']
            features['macd_histogram'] = macd_data['Histogram']
            
            # Volume features (if available)
            if 'volume' in data.columns:
                features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
                features['price_volume_corr'] = data['close'].rolling(20).corr(data['volume'])
            else:
                features['volume_sma_ratio'] = 1.0
                features['price_volume_corr'] = 0.0
            
            # Range and volatility relationships
            atr = self.technical_indicators.atr(data['high'], data['low'], data['close'], 14)
            features['atr_normalized'] = atr / data['close']
            
            # Bollinger Band position
            bb_data = self.technical_indicators.bollinger_bands(data['close'], 20, 2.0)
            features['bb_position'] = bb_data['BB_Position']
            features['bb_width'] = bb_data['BB_Width']
            
            # Higher-order moments
            features['skewness'] = returns.rolling(30).skew()
            features['kurtosis'] = returns.rolling(30).kurt()
            
            # Regime persistence indicators
            features['volatility_regime'] = (features['volatility_percentile'] > 0.7).astype(int)
            features['trend_regime'] = (np.abs(features['trend_strength']) > 0.02).astype(int)
            
            # Clean and normalize features
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.ffill().fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating regime features: {e}")
            return pd.DataFrame(index=data.index)
    
    def _detect_regimes_hmm(self, features: pd.DataFrame) -> pd.Series:
        """Detect regimes using Hidden Markov Model"""
        try:
            # Select key features for HMM
            key_features = [
                'volatility_percentile', 'trend_strength', 'rsi_momentum',
                'macd_signal', 'bb_position', 'atr_normalized'
            ]
            
            available_features = [f for f in key_features if f in features.columns]
            if len(available_features) == 0:
                self.logger.warning("No features available for HMM regime detection")
                return pd.Series(0, index=features.index)
            
            X = features[available_features].values
            
            # Handle missing data
            if np.any(np.isnan(X)):
                X = np.nan_to_num(X)
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit HMM with 4 states (regimes)
            n_regimes = min(4, len(MarketRegime))
            self.hmm_model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                random_state=42,
                n_iter=100
            )
            
            # Fit model
            self.hmm_model.fit(X_scaled)
            
            # Predict regime sequence
            regime_sequence = self.hmm_model.predict(X_scaled)
            
            # Smooth regime transitions
            smoothed_regimes = self._smooth_regime_transitions(regime_sequence)
            
            self.fitted = True
            return pd.Series(smoothed_regimes, index=features.index)
            
        except Exception as e:
            self.logger.warning(f"HMM regime detection failed: {e}")
            # Fallback to simple volatility-based regime detection
            return self._simple_regime_detection(features)
    
    def _simple_regime_detection(self, features: pd.DataFrame) -> pd.Series:
        """Simple fallback regime detection based on volatility and trend"""
        regimes = pd.Series(0, index=features.index)
        
        try:
            if 'volatility_percentile' in features.columns and 'trend_strength' in features.columns:
                vol_percentile = features['volatility_percentile']
                trend_strength = features['trend_strength']
                
                # High volatility regime
                high_vol = vol_percentile > 0.8
                
                # Trending regimes
                strong_up_trend = trend_strength > 0.03
                strong_down_trend = trend_strength < -0.03
                
                # Assign regimes
                regimes[high_vol] = 3  # High volatility
                regimes[strong_up_trend & ~high_vol] = 1  # Trending up
                regimes[strong_down_trend & ~high_vol] = 2  # Trending down
                # Default (0) is ranging/consolidation
                
        except Exception as e:
            self.logger.warning(f"Simple regime detection failed: {e}")
        
        return regimes
    
    def _smooth_regime_transitions(self, regimes: np.ndarray) -> np.ndarray:
        """Smooth regime transitions to reduce noise"""
        smoothed = regimes.copy()
        
        for i in range(self.regime_memory, len(regimes)):
            # Check if current regime is different from recent majority
            recent_regimes = regimes[i-self.regime_memory:i]
            majority_regime = np.bincount(recent_regimes).argmax()
            
            # If current regime is isolated, replace with majority
            if regimes[i] != majority_regime:
                # Check if next few are also different
                look_ahead = min(3, len(regimes) - i - 1)
                if look_ahead > 0:
                    future_regimes = regimes[i:i+look_ahead+1]
                    if np.sum(future_regimes == regimes[i]) == 1:  # Isolated
                        smoothed[i] = majority_regime
        
        return smoothed
    
    def _map_labels_to_regimes(self, 
                              labels: pd.Series, 
                              features: pd.DataFrame) -> pd.Series:
        """Map numeric labels to market regime enums"""
        regime_mapping = {}
        
        try:
            # Analyze characteristics of each numeric label
            for label in labels.unique():
                label_mask = labels == label
                label_features = features[label_mask]
                
                if len(label_features) == 0:
                    continue
                
                # Calculate average characteristics
                avg_volatility = label_features['volatility_percentile'].mean() if 'volatility_percentile' in label_features else 0.5
                avg_trend = label_features['trend_strength'].mean() if 'trend_strength' in label_features else 0.0
                
                # Map to regime based on characteristics
                if avg_volatility > 0.8:
                    regime_mapping[label] = MarketRegime.HIGH_VOLATILITY
                elif avg_volatility < 0.2:
                    regime_mapping[label] = MarketRegime.LOW_VOLATILITY
                elif avg_trend > 0.02:
                    regime_mapping[label] = MarketRegime.TRENDING_UP
                elif avg_trend < -0.02:
                    regime_mapping[label] = MarketRegime.TRENDING_DOWN
                else:
                    regime_mapping[label] = MarketRegime.RANGING
            
            # Map labels to regimes
            mapped_regimes = labels.map(regime_mapping)
            
            # Fill any unmapped values with RANGING
            mapped_regimes = mapped_regimes.fillna(MarketRegime.RANGING)
            
            return mapped_regimes
            
        except Exception as e:
            self.logger.warning(f"Error mapping labels to regimes: {e}")
            return pd.Series(MarketRegime.RANGING, index=labels.index)
    
    def _calculate_regime_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime probabilities using Gaussian Mixture Model"""
        try:
            # Select features for probability calculation
            key_features = ['volatility_percentile', 'trend_strength', 'rsi_momentum']
            available_features = [f for f in key_features if f in features.columns]
            
            if len(available_features) == 0:
                # Return uniform probabilities
                n_regimes = len(MarketRegime)
                prob_data = np.ones((len(features), n_regimes)) / n_regimes
                return pd.DataFrame(
                    prob_data, 
                    index=features.index,
                    columns=[regime.value for regime in MarketRegime]
                )
            
            X = features[available_features].values
            X = np.nan_to_num(X)
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit Gaussian Mixture Model
            n_regimes = min(len(MarketRegime), 6)
            self.gmm_model = GaussianMixture(
                n_components=n_regimes,
                random_state=42,
                max_iter=100
            )
            
            self.gmm_model.fit(X_scaled)
            
            # Calculate probabilities
            probabilities = self.gmm_model.predict_proba(X_scaled)
            
            # Create DataFrame with regime names
            regime_names = [regime.value for regime in list(MarketRegime)[:n_regimes]]
            prob_df = pd.DataFrame(
                probabilities,
                index=features.index,
                columns=regime_names
            )
            
            return prob_df
            
        except Exception as e:
            self.logger.warning(f"Error calculating regime probabilities: {e}")
            # Return uniform probabilities as fallback
            n_regimes = len(MarketRegime)
            prob_data = np.ones((len(features), n_regimes)) / n_regimes
            return pd.DataFrame(
                prob_data,
                index=features.index,
                columns=[regime.value for regime in MarketRegime]
            )
    
    def _analyze_regime_characteristics(self,
                                      data: pd.DataFrame,
                                      regimes: pd.Series,
                                      features: pd.DataFrame) -> Dict[MarketRegime, RegimeCharacteristics]:
        """Analyze characteristics of each detected regime"""
        characteristics = {}
        
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_data = data[regime_mask]
            regime_features = features[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            try:
                # Calculate regime characteristics
                returns = regime_data['close'].pct_change().dropna()
                
                volatility_percentile = regime_features['volatility_percentile'].mean() if 'volatility_percentile' in regime_features else 0.5
                trend_strength = abs(regime_features['trend_strength'].mean()) if 'trend_strength' in regime_features else 0.0
                momentum_strength = abs(regime_features['rsi_momentum'].mean()) if 'rsi_momentum' in regime_features else 0.0
                
                # Mean reversion tendency (autocorrelation of returns)
                if len(returns) > 10:
                    autocorr = returns.autocorr(lag=1)
                    mean_reversion = -autocorr if not np.isnan(autocorr) else 0.0
                else:
                    mean_reversion = 0.0
                
                # Regime persistence (how long regime lasts on average)
                regime_changes = regimes.diff() != 0
                regime_durations = []
                current_duration = 1
                
                for i in range(1, len(regimes)):
                    if regimes.iloc[i] == regime and not regime_changes.iloc[i]:
                        current_duration += 1
                    else:
                        if regimes.iloc[i-1] == regime:
                            regime_durations.append(current_duration)
                        current_duration = 1
                
                persistence = np.mean(regime_durations) if regime_durations else 1.0
                
                # Recommended indicators based on regime characteristics
                if volatility_percentile > 0.7:
                    recommended = ['ATR', 'Bollinger_Bands', 'MACD']
                elif trend_strength > 0.02:
                    recommended = ['Moving_Averages', 'MACD', 'RSI']
                else:
                    recommended = ['RSI', 'Stochastic', 'Bollinger_Bands']
                
                characteristics[regime] = RegimeCharacteristics(
                    regime=regime,
                    volatility_percentile=volatility_percentile,
                    trend_strength=trend_strength,
                    momentum_strength=momentum_strength,
                    mean_reversion_tendency=mean_reversion,
                    regime_persistence=persistence,
                    recommended_indicators=recommended
                )
                
            except Exception as e:
                self.logger.warning(f"Error analyzing regime {regime}: {e}")
        
        return characteristics
    
    def _calculate_transition_matrix(self, regimes: pd.Series) -> np.ndarray:
        """Calculate regime transition probability matrix"""
        try:
            unique_regimes = list(regimes.unique())
            n_regimes = len(unique_regimes)
            
            if n_regimes == 0:
                return np.array([])
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regimes) - 1):
                current_regime = regimes.iloc[i]
                next_regime = regimes.iloc[i + 1]
                
                current_idx = unique_regimes.index(current_regime)
                next_idx = unique_regimes.index(next_regime)
                
                transition_matrix[current_idx, next_idx] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1)
            for i in range(n_regimes):
                if row_sums[i] > 0:
                    transition_matrix[i] = transition_matrix[i] / row_sums[i]
            
            return transition_matrix
            
        except Exception as e:
            self.logger.warning(f"Error calculating transition matrix: {e}")
            return np.array([])
    
    def _calculate_regime_stability(self, regimes: pd.Series) -> float:
        """Calculate overall regime stability (persistence)"""
        try:
            if len(regimes) <= 1:
                return 1.0
            
            # Calculate regime changes
            regime_changes = (regimes.diff() != 0).sum()
            stability = 1.0 - (regime_changes / len(regimes))
            
            return max(0.0, stability)
            
        except Exception as e:
            self.logger.warning(f"Error calculating regime stability: {e}")
            return 0.5