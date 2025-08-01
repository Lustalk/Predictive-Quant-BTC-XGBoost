"""
Multi-Objective Functions for Technical Indicator Optimization
===========================================================

Comprehensive objective functions that balance multiple trading performance metrics:
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown (risk management)
- Win Rate (consistency)
- Profit Factor (efficiency)
- Calmar Ratio (drawdown-adjusted returns)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..technical_indicators import TechnicalIndicators
from ...utils.logging import get_logger


class ObjectiveFunctions:
    """
    Collection of objective functions for optimizing trading strategies.
    
    Each function takes market data and indicator parameters, then returns
    a performance metric to optimize (higher is better).
    """
    
    def __init__(self):
        """Initialize objective functions calculator."""
        self.logger = get_logger().get_logger()
        self.technical_indicators = TechnicalIndicators()
    
    def sharpe_ratio_objective(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Optimize for Sharpe ratio (risk-adjusted returns).
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            
        Returns:
            Sharpe ratio (higher is better)
        """
        try:
            # Generate trading signals based on indicators
            signals = self._generate_signals(data, params)
            
            # Calculate strategy returns
            returns = self._calculate_strategy_returns(data, signals)
            
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            # Calculate Sharpe ratio (assuming 252 trading days per year)
            mean_return = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
            risk_free_rate = 0.02  # 2% risk-free rate
            
            sharpe_ratio = (mean_return - risk_free_rate) / volatility
            
            return sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def max_drawdown_objective(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Optimize for minimum maximum drawdown (risk management).
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            
        Returns:
            Negative maximum drawdown (higher is better)
        """
        try:
            signals = self._generate_signals(data, params)
            returns = self._calculate_strategy_returns(data, signals)
            
            if len(returns) == 0:
                return -1.0  # Worst possible drawdown
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate rolling maximum (peak)
            rolling_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Return negative drawdown (so higher is better)
            return -max_drawdown if not np.isnan(max_drawdown) else -1.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating max drawdown: {e}")
            return -1.0
    
    def win_rate_objective(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Optimize for win rate (percentage of profitable trades).
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            
        Returns:
            Win rate as decimal (0.0 to 1.0)
        """
        try:
            signals = self._generate_signals(data, params)
            returns = self._calculate_strategy_returns(data, signals)
            
            if len(returns) == 0:
                return 0.0
            
            # Calculate win rate
            profitable_trades = (returns > 0).sum()
            total_trades = len(returns[returns != 0])  # Exclude no-position periods
            
            if total_trades == 0:
                return 0.0
            
            win_rate = profitable_trades / total_trades
            return win_rate if not np.isnan(win_rate) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating win rate: {e}")
            return 0.0
    
    def profit_factor_objective(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Optimize for profit factor (gross profit / gross loss).
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            
        Returns:
            Profit factor (higher is better, >1.0 is profitable)
        """
        try:
            signals = self._generate_signals(data, params)
            returns = self._calculate_strategy_returns(data, signals)
            
            if len(returns) == 0:
                return 0.0
            
            # Separate winning and losing trades
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            
            if len(winning_trades) == 0 or len(losing_trades) == 0:
                return 0.0
            
            gross_profit = winning_trades.sum()
            gross_loss = abs(losing_trades.sum())
            
            if gross_loss == 0:
                return 10.0  # Perfect strategy (no losses)
            
            profit_factor = gross_profit / gross_loss
            return profit_factor if not np.isnan(profit_factor) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating profit factor: {e}")
            return 0.0
    
    def calmar_ratio_objective(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Optimize for Calmar ratio (annualized return / max drawdown).
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            
        Returns:
            Calmar ratio (higher is better)
        """
        try:
            signals = self._generate_signals(data, params)
            returns = self._calculate_strategy_returns(data, signals)
            
            if len(returns) == 0:
                return 0.0
            
            # Calculate annualized return
            annualized_return = returns.mean() * 252
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            if max_drawdown == 0:
                return 10.0  # Perfect strategy (no drawdown)
            
            calmar_ratio = annualized_return / max_drawdown
            return calmar_ratio if not np.isnan(calmar_ratio) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def composite_objective(self, data: pd.DataFrame, params: Dict[str, Any],
                          weights: Dict[str, float] = None) -> float:
        """
        Multi-objective composite function combining multiple performance metrics.
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            weights: Weights for different objectives
            
        Returns:
            Weighted composite score
        """
        if weights is None:
            weights = {
                'sharpe_ratio': 0.4,      # Risk-adjusted returns
                'max_drawdown': 0.3,      # Risk management
                'win_rate': 0.2,          # Consistency
                'profit_factor': 0.1      # Efficiency
            }
        
        try:
            # Calculate individual objectives
            sharpe = self.sharpe_ratio_objective(data, params)
            drawdown = self.max_drawdown_objective(data, params)
            win_rate = self.win_rate_objective(data, params)
            profit_factor = self.profit_factor_objective(data, params)
            
            # Normalize profit factor to 0-1 range (assuming max reasonable value is 5.0)
            profit_factor_normalized = min(profit_factor / 5.0, 1.0)
            
            # Calculate weighted composite score
            composite_score = (
                weights['sharpe_ratio'] * max(sharpe, 0) +      # Sharpe can be negative
                weights['max_drawdown'] * drawdown +            # Already negative drawdown
                weights['win_rate'] * win_rate +                # 0-1 range
                weights['profit_factor'] * profit_factor_normalized  # 0-1 range
            )
            
            return composite_score if not np.isnan(composite_score) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating composite objective: {e}")
            return 0.0
    
    def sortino_ratio_objective(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Optimize for Sortino ratio (downside deviation adjusted returns).
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            
        Returns:
            Sortino ratio (higher is better)
        """
        try:
            signals = self._generate_signals(data, params)
            returns = self._calculate_strategy_returns(data, signals)
            
            if len(returns) == 0:
                return 0.0
            
            # Calculate Sortino ratio
            mean_return = returns.mean() * 252  # Annualized
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0:
                return 10.0  # Perfect strategy (no negative returns)
            
            downside_deviation = negative_returns.std() * np.sqrt(252)
            risk_free_rate = 0.02
            
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
            return sortino_ratio if not np.isnan(sortino_ratio) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate trading signals using XGBoost to learn optimal rules automatically.
        
        This revolutionary approach uses machine learning to discover optimal
        trading patterns instead of hardcoded rules.
        
        Args:
            data: Historical OHLCV data
            params: XGBoost hyperparameters and indicator parameters
            
        Returns:
            Trading signals (1 = buy, -1 = sell, 0 = hold)
        """
        signals = pd.Series(0, index=data.index)
        
        try:
            # XGBoost hyperparameters from optimization
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', 6)
            learning_rate = params.get('learning_rate', 0.1)
            min_child_weight = params.get('min_child_weight', 1)
            subsample = params.get('subsample', 0.8)
            colsample_bytree = params.get('colsample_bytree', 0.8)
            
            # Technical indicator parameters
            rsi_period = params.get('rsi_period', 14)
            macd_fast = params.get('macd_fast', 12)
            macd_slow = params.get('macd_slow', 26)
            macd_signal = params.get('macd_signal', 9)
            bb_period = params.get('bb_period', 20)
            bb_std = params.get('bb_std', 2.0)
            
            # Signal generation parameters
            prediction_threshold = params.get('prediction_threshold', 0.55)
            lookforward_periods = params.get('lookforward_periods', 4)
            target_threshold = params.get('target_threshold', 0.001)
            momentum_lookback = params.get('momentum_lookback', 5)
            
            # Calculate comprehensive technical indicators
            features_df = self._create_ml_features(data, params)
            
            if len(features_df) < 50:  # Need minimum data for training
                return signals
            
            # Create target variable (future price direction)
            target = self._create_target_variable(data, lookforward_periods, target_threshold)
            
            # Align features and target
            min_length = min(len(features_df), len(target))
            features_df = features_df.iloc[:min_length]
            target = target.iloc[:min_length]
            
            # Remove rows with NaN
            valid_mask = ~(features_df.isna().any(axis=1) | target.isna())
            features_clean = features_df[valid_mask]
            target_clean = target[valid_mask]
            
            if len(features_clean) < 30:  # Need minimum samples
                return signals
            
            # Train XGBoost model
            try:
                import xgboost as xgb
                
                # Split data for training (use first 70% for training, rest for signals)
                split_idx = int(len(features_clean) * 0.7)
                
                X_train = features_clean.iloc[:split_idx]
                y_train = target_clean.iloc[:split_idx]
                X_predict = features_clean.iloc[split_idx:]
                
                if len(X_train) < 20 or len(X_predict) < 5:
                    return signals
                
                # Configure XGBoost model
                model = xgb.XGBClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    learning_rate=learning_rate,
                    min_child_weight=int(min_child_weight),
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42,
                    eval_metric='logloss',
                    verbosity=0
                )
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Generate predictions
                predictions = model.predict_proba(X_predict)[:, 1]  # Probability of positive class
                
                # Convert predictions to trading signals
                prediction_signals = pd.Series(0, index=X_predict.index)
                
                # Buy signals: High confidence of price increase
                prediction_signals[predictions > prediction_threshold] = 1
                
                # Sell signals: High confidence of price decrease  
                prediction_signals[predictions < (1 - prediction_threshold)] = -1
                
                # Map back to original index
                signals.loc[prediction_signals.index] = prediction_signals
                
                return signals
                
            except ImportError:
                self.logger.warning("XGBoost not available, falling back to simple momentum strategy")
                return self._fallback_momentum_strategy(data, params)
            
        except Exception as e:
            self.logger.warning(f"Error in XGBoost signal generation: {e}")
            return self._fallback_momentum_strategy(data, params)
    
    def _create_ml_features(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive features for XGBoost training."""
        features = pd.DataFrame(index=data.index)
        
        try:
            # Extract indicator parameters
            rsi_period = params.get('rsi_period', 14)
            macd_fast = params.get('macd_fast', 12)
            macd_slow = params.get('macd_slow', 26)
            macd_signal = params.get('macd_signal', 9)
            bb_period = params.get('bb_period', 20)
            bb_std = params.get('bb_std', 2.0)
            
            # Price features
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['price_sma_ratio'] = data['close'] / data['close'].rolling(20).mean()
            features['high_low_ratio'] = data['high'] / data['low']
            features['close_open_ratio'] = data['close'] / data['open']
            
            # Volume features
            features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            features['price_volume_corr'] = data['close'].rolling(20).corr(data['volume'])
            
            # Technical indicators
            rsi = self.technical_indicators.rsi(data['close'], rsi_period)
            features['rsi'] = rsi
            features['rsi_oversold'] = (rsi < 30).astype(int)
            features['rsi_overbought'] = (rsi > 70).astype(int)
            features['rsi_momentum'] = rsi.diff()
            
            # MACD features
            macd_data = self.technical_indicators.macd(data['close'], macd_fast, macd_slow, macd_signal)
            features['macd'] = macd_data['MACD']
            features['macd_signal'] = macd_data['Signal']
            features['macd_histogram'] = macd_data['Histogram']
            features['macd_crossover'] = (macd_data['MACD'] > macd_data['Signal']).astype(int)
            
            # Bollinger Bands features
            bb_data = self.technical_indicators.bollinger_bands(data['close'], bb_period, bb_std)
            features['bb_position'] = bb_data['BB_Position']
            features['bb_width'] = bb_data['BB_Width']
            features['bb_squeeze'] = (bb_data['BB_Width'] < bb_data['BB_Width'].rolling(20).mean() * 0.5).astype(int)
            
            # Momentum features
            features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
            features['momentum_10'] = data['close'] / data['close'].shift(10) - 1
            features['momentum_20'] = data['close'] / data['close'].shift(20) - 1
            
            # Volatility features
            features['volatility'] = data['close'].pct_change().rolling(20).std()
            features['atr'] = self.technical_indicators.atr(data['high'], data['low'], data['close'], 14)
            features['atr_normalized'] = features['atr'] / data['close']
            
            # Moving averages
            features['sma_5'] = data['close'].rolling(5).mean()
            features['sma_20'] = data['close'].rolling(20).mean()
            features['sma_50'] = data['close'].rolling(50).mean()
            features['price_above_sma5'] = (data['close'] > features['sma_5']).astype(int)
            features['price_above_sma20'] = (data['close'] > features['sma_20']).astype(int)
            features['sma_trend'] = (features['sma_5'] > features['sma_20']).astype(int)
            
            # Market structure features
            features['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
            features['higher_low'] = (data['low'] > data['low'].shift(1)).astype(int)
            features['trend_strength'] = features['higher_high'] + features['higher_low']
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error creating ML features: {e}")
            return pd.DataFrame(index=data.index)
    
    def _create_target_variable(self, data: pd.DataFrame, lookforward_periods: int = 4, threshold: float = 0.001) -> pd.Series:
        """Create target variable for XGBoost training (future price direction)."""
        try:
            # Calculate future returns
            future_returns = data['close'].pct_change(periods=lookforward_periods).shift(-lookforward_periods)
            
            # Create binary target: 1 if price goes up significantly, 0 otherwise
            target = (future_returns > threshold).astype(int)
            
            return target
            
        except Exception as e:
            self.logger.warning(f"Error creating target variable: {e}")
            return pd.Series(0, index=data.index)
    
    def _fallback_momentum_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Fallback momentum strategy if XGBoost fails."""
        signals = pd.Series(0, index=data.index)
        
        try:
            # Simple momentum strategy
            short_ma = data['close'].rolling(5).mean()
            long_ma = data['close'].rolling(20).mean()
            rsi = self.technical_indicators.rsi(data['close'], 14)
            
            # Buy: Short MA > Long MA and RSI not overbought
            buy_condition = (short_ma > long_ma) & (rsi < 75)
            
            # Sell: Short MA < Long MA and RSI not oversold
            sell_condition = (short_ma < long_ma) & (rsi > 25)
            
            signals[buy_condition] = 1
            signals[sell_condition] = -1
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"Error in fallback strategy: {e}")
            return signals
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate strategy returns based on trading signals.
        
        Args:
            data: Historical OHLCV data
            signals: Trading signals
            
        Returns:
            Strategy returns
        """
        try:
            # Calculate price returns
            price_returns = data['close'].pct_change().fillna(0)
            
            # Apply signals with lag to avoid lookahead bias
            lagged_signals = signals.shift(1).fillna(0)
            
            # Calculate strategy returns
            strategy_returns = price_returns * lagged_signals
            
            # Apply transaction costs (0.1% per trade)
            transaction_cost = 0.001
            signal_changes = lagged_signals.diff().abs()
            costs = signal_changes * transaction_cost
            
            # Subtract transaction costs
            strategy_returns = strategy_returns - costs
            
            return strategy_returns.fillna(0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating strategy returns: {e}")
            return pd.Series(0, index=data.index)
    
    def calculate_performance_metrics(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for given parameters.
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            signals = self._generate_signals(data, params)
            returns = self._calculate_strategy_returns(data, signals)
            
            metrics = {
                'sharpe_ratio': self.sharpe_ratio_objective(data, params),
                'max_drawdown': -self.max_drawdown_objective(data, params),  # Convert back to positive
                'win_rate': self.win_rate_objective(data, params),
                'profit_factor': self.profit_factor_objective(data, params),
                'calmar_ratio': self.calmar_ratio_objective(data, params),
                'sortino_ratio': self.sortino_ratio_objective(data, params),
                'total_return': ((1 + returns).cumprod().iloc[-1] - 1) if len(returns) > 0 else 0.0,
                'annualized_return': returns.mean() * 252 if len(returns) > 0 else 0.0,
                'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0,
                'num_trades': (signals != 0).sum(),
                'avg_trade_return': returns[returns != 0].mean() if len(returns[returns != 0]) > 0 else 0.0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}