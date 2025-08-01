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
        Generate trading signals based on technical indicator parameters.
        
        This is a simplified signal generation logic. In practice, you would
        implement your specific trading strategy logic here.
        
        Args:
            data: Historical OHLCV data
            params: Technical indicator parameters
            
        Returns:
            Trading signals (1 = buy, -1 = sell, 0 = hold)
        """
        signals = pd.Series(0, index=data.index)
        
        try:
            # Extract parameters (example for RSI + MACD strategy)
            rsi_period = params.get('rsi_period', 14)
            rsi_oversold = params.get('rsi_oversold', 30)
            rsi_overbought = params.get('rsi_overbought', 70)
            
            macd_fast = params.get('macd_fast', 12)
            macd_slow = params.get('macd_slow', 26)
            macd_signal = params.get('macd_signal', 9)
            
            bb_period = params.get('bb_period', 20)
            bb_std = params.get('bb_std', 2.0)
            
            # Calculate indicators
            rsi = self.technical_indicators.rsi(data['close'], rsi_period)
            macd_data = self.technical_indicators.macd(data['close'], macd_fast, macd_slow, macd_signal)
            bb_data = self.technical_indicators.bollinger_bands(data['close'], bb_period, bb_std)
            
            # Generate signals (example strategy)
            # Buy signal: RSI oversold AND MACD bullish AND price near lower BB
            buy_condition = (
                (rsi < rsi_oversold) & 
                (macd_data['MACD'] > macd_data['Signal']) &
                (data['close'] < bb_data['BB_Lower'] * 1.02)  # Within 2% of lower band
            )
            
            # Sell signal: RSI overbought AND MACD bearish AND price near upper BB
            sell_condition = (
                (rsi > rsi_overbought) & 
                (macd_data['MACD'] < macd_data['Signal']) &
                (data['close'] > bb_data['BB_Upper'] * 0.98)  # Within 2% of upper band
            )
            
            signals[buy_condition] = 1
            signals[sell_condition] = -1
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"Error generating signals: {e}")
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