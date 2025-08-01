"""
Parameter Search Spaces for Technical Indicator Optimization
==========================================================

Defines intelligent search spaces for Bayesian optimization of technical indicators.
Each space is carefully crafted based on:
- Traditional indicator ranges used in quantitative finance
- Market characteristics of cryptocurrency trading
- Computational efficiency considerations
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from skopt.space import Real, Integer, Categorical

from ...utils.logging import get_logger


class ParameterSpaces:
    """
    Comprehensive parameter search spaces for technical indicators.
    
    Each search space is optimized for Bayesian optimization efficiency
    while covering meaningful parameter ranges for Bitcoin trading.
    """
    
    def __init__(self):
        """Initialize parameter space definitions."""
        self.logger = get_logger().get_logger()
    
    def get_rsi_space(self, extended: bool = False) -> Tuple[List[Any], List[str]]:
        """
        RSI (Relative Strength Index) parameter space.
        
        Args:
            extended: If True, includes more parameters for comprehensive optimization
            
        Returns:
            Tuple of (search_space, parameter_names)
        """
        if extended:
            # Extended space for deep optimization
            search_space = [
                Integer(5, 50, name='rsi_period'),           # RSI calculation period
                Integer(20, 40, name='rsi_oversold'),        # Oversold threshold
                Integer(60, 80, name='rsi_overbought'),      # Overbought threshold
                Integer(2, 10, name='rsi_smoothing'),        # Additional smoothing
                Categorical(['simple', 'exponential'], name='rsi_ma_type')  # MA type for smoothing
            ]
            parameter_names = ['rsi_period', 'rsi_oversold', 'rsi_overbought', 'rsi_smoothing', 'rsi_ma_type']
        else:
            # Standard space for quick optimization
            search_space = [
                Integer(7, 35, name='rsi_period'),           # RSI calculation period
                Integer(25, 35, name='rsi_oversold'),        # Oversold threshold
                Integer(65, 75, name='rsi_overbought')       # Overbought threshold
            ]
            parameter_names = ['rsi_period', 'rsi_oversold', 'rsi_overbought']
        
        return search_space, parameter_names
    
    def get_macd_space(self, extended: bool = False) -> Tuple[List[Any], List[str]]:
        """
        MACD (Moving Average Convergence Divergence) parameter space.
        
        Args:
            extended: If True, includes more parameter combinations
            
        Returns:
            Tuple of (search_space, parameter_names)
        """
        if extended:
            search_space = [
                Integer(8, 20, name='macd_fast'),            # Fast EMA period
                Integer(20, 40, name='macd_slow'),           # Slow EMA period
                Integer(6, 15, name='macd_signal'),          # Signal line period
                Real(0.5, 2.0, name='macd_threshold'),       # Signal threshold multiplier
                Categorical(['standard', 'zero_lag'], name='macd_type')  # MACD variant
            ]
            parameter_names = ['macd_fast', 'macd_slow', 'macd_signal', 'macd_threshold', 'macd_type']
        else:
            search_space = [
                Integer(8, 18, name='macd_fast'),            # Fast EMA period
                Integer(22, 35, name='macd_slow'),           # Slow EMA period
                Integer(7, 12, name='macd_signal')           # Signal line period
            ]
            parameter_names = ['macd_fast', 'macd_slow', 'macd_signal']
        
        return search_space, parameter_names
    
    def get_bollinger_bands_space(self, extended: bool = False) -> Tuple[List[Any], List[str]]:
        """
        Bollinger Bands parameter space.
        
        Args:
            extended: If True, includes additional parameters
            
        Returns:
            Tuple of (search_space, parameter_names)
        """
        if extended:
            search_space = [
                Integer(10, 50, name='bb_period'),           # Moving average period
                Real(1.5, 3.0, name='bb_std'),               # Standard deviation multiplier
                Real(0.01, 0.1, name='bb_squeeze_threshold'), # Squeeze detection
                Categorical(['sma', 'ema', 'wma'], name='bb_ma_type'),  # MA type
                Real(0.8, 1.2, name='bb_band_ratio')         # Band ratio for signals
            ]
            parameter_names = ['bb_period', 'bb_std', 'bb_squeeze_threshold', 'bb_ma_type', 'bb_band_ratio']
        else:
            search_space = [
                Integer(12, 30, name='bb_period'),           # Moving average period
                Real(1.8, 2.5, name='bb_std')                # Standard deviation multiplier
            ]
            parameter_names = ['bb_period', 'bb_std']
        
        return search_space, parameter_names
    
    def get_moving_averages_space(self, extended: bool = False) -> Tuple[List[Any], List[str]]:
        """
        Moving Averages parameter space for crossover strategies.
        
        Args:
            extended: If True, includes multiple MA combinations
            
        Returns:
            Tuple of (search_space, parameter_names)
        """
        if extended:
            search_space = [
                Integer(3, 25, name='ma_fast'),              # Fast MA period
                Integer(25, 100, name='ma_slow'),            # Slow MA period
                Integer(50, 200, name='ma_trend'),           # Trend filter MA
                Categorical(['sma', 'ema', 'wma'], name='ma_fast_type'),  # Fast MA type
                Categorical(['sma', 'ema', 'wma'], name='ma_slow_type'),  # Slow MA type
                Real(0.001, 0.02, name='ma_filter'),         # Noise filter
            ]
            parameter_names = ['ma_fast', 'ma_slow', 'ma_trend', 'ma_fast_type', 'ma_slow_type', 'ma_filter']
        else:
            search_space = [
                Integer(5, 20, name='ma_fast'),              # Fast MA period
                Integer(30, 80, name='ma_slow')              # Slow MA period
            ]
            parameter_names = ['ma_fast', 'ma_slow']
        
        return search_space, parameter_names
    
    def get_stochastic_space(self, extended: bool = False) -> Tuple[List[Any], List[str]]:
        """
        Stochastic Oscillator parameter space.
        
        Args:
            extended: If True, includes additional smoothing parameters
            
        Returns:
            Tuple of (search_space, parameter_names)
        """
        if extended:
            search_space = [
                Integer(5, 30, name='stoch_k_period'),       # %K period
                Integer(1, 10, name='stoch_d_period'),       # %D smoothing
                Integer(1, 5, name='stoch_smooth'),          # Additional smoothing
                Integer(10, 30, name='stoch_oversold'),      # Oversold level
                Integer(70, 90, name='stoch_overbought')     # Overbought level
            ]
            parameter_names = ['stoch_k_period', 'stoch_d_period', 'stoch_smooth', 'stoch_oversold', 'stoch_overbought']
        else:
            search_space = [
                Integer(8, 21, name='stoch_k_period'),       # %K period
                Integer(2, 6, name='stoch_d_period')         # %D smoothing
            ]
            parameter_names = ['stoch_k_period', 'stoch_d_period']
        
        return search_space, parameter_names
    
    def get_atr_space(self, extended: bool = False) -> Tuple[List[Any], List[str]]:
        """
        Average True Range parameter space.
        
        Args:
            extended: If True, includes volatility-based parameters
            
        Returns:
            Tuple of (search_space, parameter_names)
        """
        if extended:
            search_space = [
                Integer(7, 35, name='atr_period'),           # ATR calculation period
                Real(1.0, 4.0, name='atr_multiplier'),       # ATR multiplier for stops
                Real(0.5, 2.0, name='atr_entry_threshold'),  # Entry threshold
                Real(1.5, 3.5, name='atr_exit_threshold')    # Exit threshold
            ]
            parameter_names = ['atr_period', 'atr_multiplier', 'atr_entry_threshold', 'atr_exit_threshold']
        else:
            search_space = [
                Integer(10, 25, name='atr_period'),          # ATR calculation period
                Real(1.5, 3.0, name='atr_multiplier')        # ATR multiplier
            ]
            parameter_names = ['atr_period', 'atr_multiplier']
        
        return search_space, parameter_names
    
    def get_xgboost_strategy_space(self) -> Tuple[List[Any], List[str]]:
        """
        XGBoost-powered strategy space for machine learning optimization.
        
        This revolutionary approach optimizes XGBoost hyperparameters and
        lets the model learn optimal trading rules automatically.
        
        Returns:
            Tuple of (search_space, parameter_names)
        """
        search_space = [
            # XGBoost hyperparameters (core ML optimization)
            Integer(50, 300, name='n_estimators'),          # Number of trees
            Integer(3, 10, name='max_depth'),               # Tree depth
            Real(0.01, 0.3, name='learning_rate'),          # Learning rate
            Integer(1, 7, name='min_child_weight'),         # Minimum samples per leaf
            Real(0.6, 1.0, name='subsample'),               # Subsample ratio
            Real(0.6, 1.0, name='colsample_bytree'),        # Feature subsample ratio
            
            # Signal generation parameters
            Real(0.52, 0.8, name='prediction_threshold'),   # Prediction confidence threshold
            Integer(2, 8, name='lookforward_periods'),      # Future periods to predict
            
            # Technical indicator parameters for feature creation
            Integer(9, 25, name='rsi_period'),              # RSI period
            Integer(8, 18, name='macd_fast'),               # MACD fast EMA
            Integer(22, 35, name='macd_slow'),              # MACD slow EMA
            Integer(7, 12, name='macd_signal'),             # MACD signal line
            Integer(15, 30, name='bb_period'),              # Bollinger Bands period
            Real(1.8, 2.5, name='bb_std'),                  # Bollinger Bands std dev
            
            # Feature engineering parameters
            Real(0.0005, 0.005, name='target_threshold'),   # Minimum return to be considered significant
            Integer(3, 12, name='momentum_lookback'),       # Momentum calculation period
        ]
        
        parameter_names = [
            'n_estimators', 'max_depth', 'learning_rate', 'min_child_weight', 
            'subsample', 'colsample_bytree',
            'prediction_threshold', 'lookforward_periods',
            'rsi_period', 'macd_fast', 'macd_slow', 'macd_signal',
            'bb_period', 'bb_std',
            'target_threshold', 'momentum_lookback'
        ]
        
        return search_space, parameter_names
    
    def get_composite_strategy_space(self) -> Tuple[List[Any], List[str]]:
        """
        Legacy composite strategy (kept for backwards compatibility).
        
        DEPRECATED: Use get_xgboost_strategy_space() for better results.
        """
        # Return XGBoost strategy space as the new default
        return self.get_xgboost_strategy_space()
    
    def get_regime_specific_spaces(self) -> Dict[str, Tuple[List[Any], List[str]]]:
        """
        Get parameter spaces optimized for different market regimes.
        
        Returns:
            Dictionary mapping regime names to (search_space, parameter_names)
        """
        spaces = {}
        
        # Trending market parameters (momentum-focused)
        spaces['trending'] = (
            [
                Integer(5, 15, name='rsi_period'),           # Shorter RSI for momentum
                Integer(6, 14, name='macd_fast'),            # Faster MACD
                Integer(18, 28, name='macd_slow'),
                Integer(5, 9, name='macd_signal'),
                Integer(10, 20, name='bb_period'),           # Shorter BB period
                Real(1.5, 2.2, name='bb_std'),
                Integer(5, 15, name='ma_fast'),              # Faster MA crossovers
                Integer(25, 50, name='ma_slow')
            ],
            ['rsi_period', 'macd_fast', 'macd_slow', 'macd_signal', 
             'bb_period', 'bb_std', 'ma_fast', 'ma_slow']
        )
        
        # Ranging market parameters (mean reversion-focused)
        spaces['ranging'] = (
            [
                Integer(14, 30, name='rsi_period'),          # Longer RSI for ranges
                Integer(12, 20, name='macd_fast'),           # Standard MACD
                Integer(26, 40, name='macd_slow'),
                Integer(9, 15, name='macd_signal'),
                Integer(20, 40, name='bb_period'),           # Longer BB for ranges
                Real(2.0, 3.0, name='bb_std'),               # Wider bands
                Integer(20, 40, name='ma_fast'),             # Slower MA crossovers
                Integer(50, 100, name='ma_slow')
            ],
            ['rsi_period', 'macd_fast', 'macd_slow', 'macd_signal',
             'bb_period', 'bb_std', 'ma_fast', 'ma_slow']
        )
        
        # High volatility parameters (risk-adjusted)
        spaces['high_volatility'] = (
            [
                Integer(7, 18, name='rsi_period'),           # Medium RSI
                Integer(8, 16, name='macd_fast'),            # Responsive MACD
                Integer(20, 32, name='macd_slow'),
                Integer(6, 10, name='macd_signal'),
                Integer(12, 25, name='bb_period'),           # Adaptive BB
                Real(2.2, 3.0, name='bb_std'),               # Wider bands for volatility
                Real(2.0, 4.0, name='atr_multiplier'),       # Higher ATR multiplier
                Real(0.01, 0.03, name='noise_filter')        # Stronger noise filter
            ],
            ['rsi_period', 'macd_fast', 'macd_slow', 'macd_signal',
             'bb_period', 'bb_std', 'atr_multiplier', 'noise_filter']
        )
        
        return spaces
    
    def get_timeframe_specific_spaces(self, timeframe: str) -> Tuple[List[Any], List[str]]:
        """
        Get parameter spaces optimized for specific timeframes.
        
        Args:
            timeframe: Timeframe string ('30m', '1h', '4h', '1d')
            
        Returns:
            Tuple of (search_space, parameter_names)
        """
        # Adjust parameters based on timeframe
        if timeframe in ['5m', '15m', '30m']:
            # Short-term parameters (faster signals)
            multiplier = 0.5
            base_periods = {'rsi': 10, 'macd_fast': 8, 'macd_slow': 20, 'bb': 15}
        elif timeframe in ['1h', '2h']:
            # Medium-term parameters
            multiplier = 1.0
            base_periods = {'rsi': 14, 'macd_fast': 12, 'macd_slow': 26, 'bb': 20}
        elif timeframe in ['4h', '6h']:
            # Long-term parameters (slower signals)
            multiplier = 1.5
            base_periods = {'rsi': 18, 'macd_fast': 16, 'macd_slow': 35, 'bb': 25}
        else:  # 1d and longer
            # Very long-term parameters
            multiplier = 2.0
            base_periods = {'rsi': 25, 'macd_fast': 20, 'macd_slow': 45, 'bb': 30}
        
        # Create adjusted search space
        search_space = [
            Integer(int(base_periods['rsi'] * 0.7), int(base_periods['rsi'] * 1.5), name='rsi_period'),
            Integer(int(base_periods['macd_fast'] * 0.7), int(base_periods['macd_fast'] * 1.3), name='macd_fast'),
            Integer(int(base_periods['macd_slow'] * 0.8), int(base_periods['macd_slow'] * 1.3), name='macd_slow'),
            Integer(int(base_periods['bb'] * 0.7), int(base_periods['bb'] * 1.4), name='bb_period'),
            Real(1.5, 2.8, name='bb_std'),
            Real(1.0 * multiplier, 3.0 * multiplier, name='atr_multiplier')
        ]
        
        parameter_names = ['rsi_period', 'macd_fast', 'macd_slow', 'bb_period', 'bb_std', 'atr_multiplier']
        
        return search_space, parameter_names
    
    def validate_parameter_constraints(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameter constraints to ensure logical combinations.
        
        Args:
            params: Parameter dictionary to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # MACD constraint: fast < slow
            if 'macd_fast' in params and 'macd_slow' in params:
                if params['macd_fast'] >= params['macd_slow']:
                    return False
            
            # RSI constraint: oversold < overbought
            if 'rsi_oversold' in params and 'rsi_overbought' in params:
                if params['rsi_oversold'] >= params['rsi_overbought']:
                    return False
            
            # Moving average constraint: fast < slow
            if 'ma_fast' in params and 'ma_slow' in params:
                if params['ma_fast'] >= params['ma_slow']:
                    return False
            
            # Stochastic constraint: oversold < overbought
            if 'stoch_oversold' in params and 'stoch_overbought' in params:
                if params['stoch_oversold'] >= params['stoch_overbought']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating parameters: {e}")
            return False
    
    def get_parameter_bounds_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of all parameter bounds for documentation.
        
        Returns:
            Dictionary with parameter bounds and descriptions
        """
        return {
            'rsi_period': {'min': 5, 'max': 50, 'description': 'RSI calculation period'},
            'rsi_oversold': {'min': 20, 'max': 40, 'description': 'RSI oversold threshold'},
            'rsi_overbought': {'min': 60, 'max': 80, 'description': 'RSI overbought threshold'},
            'macd_fast': {'min': 6, 'max': 20, 'description': 'MACD fast EMA period'},
            'macd_slow': {'min': 18, 'max': 45, 'description': 'MACD slow EMA period'},
            'macd_signal': {'min': 5, 'max': 15, 'description': 'MACD signal line period'},
            'bb_period': {'min': 10, 'max': 50, 'description': 'Bollinger Bands MA period'},
            'bb_std': {'min': 1.5, 'max': 3.0, 'description': 'Bollinger Bands std dev multiplier'},
            'ma_fast': {'min': 3, 'max': 25, 'description': 'Fast moving average period'},
            'ma_slow': {'min': 25, 'max': 200, 'description': 'Slow moving average period'},
            'atr_period': {'min': 7, 'max': 35, 'description': 'ATR calculation period'},
            'atr_multiplier': {'min': 1.0, 'max': 4.0, 'description': 'ATR multiplier for stops'}
        }