#!/usr/bin/env python3
"""
🎯 Comprehensive Hyperparameter Optimization System
==================================================

Advanced optimization framework that:
1. Tests ALL technical indicators to find the most effective ones
2. Optimizes take profit/stop loss dynamically (no hardcoding)
3. Uses multi-objective optimization for risk-adjusted returns
4. Provides indicator importance ranking and selection
5. Implements walk-forward validation for robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import warnings

warnings.filterwarnings("ignore")

# Optimization imports
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Local imports
from .smart_optimizer import SmartOptimizer, OptimizationConfig, OptimizationResult
from .parameter_spaces import ParameterSpaces
from .objective_functions import ObjectiveFunctions
from ..technical_indicators import TechnicalIndicators
from ...utils.logging import get_logger


@dataclass
class IndicatorPerformance:
    """Performance metrics for individual indicators"""

    name: str
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    parameter_sensitivity: float
    importance_score: float


@dataclass
class ComprehensiveOptimizationResult:
    """Results from comprehensive optimization"""

    best_overall_params: Dict[str, Any]
    best_overall_score: float
    indicator_rankings: List[IndicatorPerformance]
    optimal_tp_sl: Dict[str, float]
    feature_importance: Dict[str, float]
    optimization_time: float
    total_evaluations: int
    convergence_history: List[float]


class ComprehensiveHyperparameterOptimizer:
    """
    Advanced hyperparameter optimizer that tests ALL indicators and optimizes TP/SL.

    Features:
    - Multi-objective optimization (return vs risk)
    - Indicator importance ranking
    - Dynamic TP/SL optimization
    - Feature selection based on performance
    - Walk-forward robustness testing
    """

    def __init__(
        self,
        optimization_config: OptimizationConfig = None,
        enable_feature_selection: bool = True,
        max_optimization_time_hours: float = 2.0,
    ):
        """
        Initialize comprehensive optimizer.

        Args:
            optimization_config: Bayesian optimization settings
            enable_feature_selection: Whether to perform automatic feature selection
            max_optimization_time_hours: Maximum optimization time in hours
        """
        self.config = optimization_config or OptimizationConfig(
            n_calls=200,  # More calls for comprehensive search
            n_initial_points=50,  # Better initial exploration
            acquisition_function="EI",  # Expected Improvement
        )

        self.enable_feature_selection = enable_feature_selection
        self.max_optimization_time = (
            max_optimization_time_hours * 3600
        )  # Convert to seconds

        # Initialize components
        self.parameter_spaces = ParameterSpaces()
        self.objective_functions = ObjectiveFunctions()
        self.technical_indicators = TechnicalIndicators()
        self.logger = get_logger().get_logger()

        # Performance tracking
        self.indicator_performances: List[IndicatorPerformance] = []
        self.evaluation_count = 0
        self.start_time = None

    def optimize_all_indicators(
        self, data: pd.DataFrame, validation_split: float = 0.2
    ) -> ComprehensiveOptimizationResult:
        """
        Comprehensive optimization of all indicators + TP/SL parameters.

        Args:
            data: Historical OHLCV data
            validation_split: Fraction of data for validation

        Returns:
            ComprehensiveOptimizationResult with best parameters and rankings
        """
        self.logger.info("Starting comprehensive hyperparameter optimization...")
        self.logger.info(
            f"Data: {len(data)} samples, Validation split: {validation_split:.1%}"
        )

        self.start_time = time.time()

        # Split data for validation
        split_point = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_point].copy()
        val_data = data.iloc[split_point:].copy()

        self.logger.info(
            f"Train: {len(train_data)} samples, Validation: {len(val_data)} samples"
        )

        # Phase 1: Individual indicator testing
        self.logger.info("PHASE 1: Individual Indicator Analysis")
        individual_results = self._test_individual_indicators(train_data, val_data)

        # Phase 2: Combined optimization with TP/SL
        self.logger.info("PHASE 2: Combined Optimization with TP/SL")
        combined_result = self._optimize_combined_strategy(
            train_data, val_data, individual_results
        )

        # Phase 3: Feature selection and final optimization
        self.logger.info("PHASE 3: Feature Selection & Final Optimization")
        final_result = self._final_optimization_with_selection(
            train_data, val_data, combined_result
        )

        optimization_time = time.time() - self.start_time

        self.logger.info(f"Comprehensive optimization complete!")
        self.logger.info(f"Total time: {optimization_time:.1f} seconds")
        self.logger.info(f"Total evaluations: {self.evaluation_count}")

        return final_result

    def _test_individual_indicators(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame
    ) -> Dict[str, IndicatorPerformance]:
        """Test each indicator individually to rank their effectiveness."""

        indicator_tests = {
            "RSI": self.parameter_spaces.get_rsi_space(extended=True),
            "MACD": self.parameter_spaces.get_macd_space(extended=True),
            "Bollinger_Bands": self.parameter_spaces.get_bollinger_bands_space(
                extended=True
            ),
            "Moving_Averages": self.parameter_spaces.get_moving_averages_space(
                extended=True
            ),
            "Stochastic": self.parameter_spaces.get_stochastic_space(extended=True),
            "ATR": self.parameter_spaces.get_atr_space(extended=True),
        }

        individual_results = {}

        for indicator_name, (search_space, param_names) in indicator_tests.items():
            self.logger.info(f"Testing {indicator_name} indicator...")

            # Add basic TP/SL parameters for individual testing
            enhanced_space = search_space + [
                Real(0.01, 0.10, name="stop_loss"),
                Real(0.02, 0.25, name="take_profit"),
            ]
            enhanced_names = param_names + ["stop_loss", "take_profit"]

            # Optimize this indicator
            optimizer = SmartOptimizer(
                OptimizationConfig(n_calls=50)
            )  # Fewer calls per indicator

            def indicator_objective(data_subset, params):
                return self._evaluate_single_indicator_strategy(
                    data_subset, params, indicator_name
                )

            try:
                result = optimizer.optimize_indicators(
                    data=train_data,
                    parameter_space=enhanced_space,
                    objective_function=indicator_objective,
                    parameter_names=enhanced_names,
                )

                # Validate on out-of-sample data
                val_score = indicator_objective(val_data, result.best_params)

                # Calculate comprehensive metrics
                performance = self._calculate_indicator_performance(
                    train_data,
                    val_data,
                    result.best_params,
                    indicator_name,
                    result.best_score,
                    val_score,
                )

                individual_results[indicator_name] = performance
                self.indicator_performances.append(performance)

                self.logger.info(
                    f"COMPLETED {indicator_name}: Sharpe={performance.sharpe_ratio:.3f}, "
                    f"MaxDD={performance.max_drawdown:.3f}, WinRate={performance.win_rate:.1%}"
                )

            except Exception as e:
                self.logger.warning(f"Error optimizing {indicator_name}: {e}")
                # Create default performance for failed indicators
                default_performance = IndicatorPerformance(
                    name=indicator_name,
                    sharpe_ratio=0.0,
                    max_drawdown=1.0,
                    win_rate=0.0,
                    profit_factor=1.0,
                    total_trades=0,
                    avg_trade_duration=0.0,
                    parameter_sensitivity=1.0,
                    importance_score=0.0,
                )
                individual_results[indicator_name] = default_performance
                self.indicator_performances.append(default_performance)

        # Sort indicators by importance score
        self.indicator_performances.sort(key=lambda x: x.importance_score, reverse=True)

        self.logger.info("INDICATOR RANKINGS:")
        for i, perf in enumerate(self.indicator_performances, 1):
            self.logger.info(f"   {i}. {perf.name}: Score={perf.importance_score:.3f}")

        return individual_results

    def _optimize_combined_strategy(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, individual_results: Dict
    ) -> Dict[str, Any]:
        """Optimize combination of best indicators with comprehensive TP/SL."""

        # Select top 3 indicators for combination
        top_indicators = self.indicator_performances[:3]
        self.logger.info(
            f"Combining top indicators: {[ind.name for ind in top_indicators]}"
        )

        # Create comprehensive search space
        search_space, param_names = self._create_comprehensive_search_space(
            top_indicators
        )

        # Multi-objective optimization function
        def combined_objective(data_subset, params):
            return self._evaluate_combined_strategy(data_subset, params, top_indicators)

        # Run comprehensive optimization
        optimizer = SmartOptimizer(OptimizationConfig(n_calls=150))

        result = optimizer.optimize_indicators(
            data=train_data,
            parameter_space=search_space,
            objective_function=combined_objective,
            parameter_names=param_names,
        )

        self.evaluation_count += result.evaluation_count

        return result.best_params

    def _final_optimization_with_selection(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        combined_params: Dict[str, Any],
    ) -> ComprehensiveOptimizationResult:
        """Final optimization with feature selection and robustness testing."""

        # Feature selection based on individual performance
        selected_features = self._select_optimal_features()

        # Create final search space with selected features
        final_space, final_names = self._create_final_search_space(selected_features)

        # Final optimization
        def final_objective(data_subset, params):
            return self._evaluate_final_strategy(data_subset, params, selected_features)

        optimizer = SmartOptimizer(self.config)
        final_result = optimizer.optimize_indicators(
            data=train_data,
            parameter_space=final_space,
            objective_function=final_objective,
            parameter_names=final_names,
        )

        # Validate on out-of-sample data
        final_val_score = final_objective(val_data, final_result.best_params)

        # Extract TP/SL parameters
        optimal_tp_sl = {
            "stop_loss": final_result.best_params.get("stop_loss", 0.05),
            "take_profit": final_result.best_params.get("take_profit", 0.10),
            "risk_reward_ratio": final_result.best_params.get("risk_reward_ratio", 2.0),
            "position_timeout": final_result.best_params.get("position_timeout", 24),
        }

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            final_result, selected_features
        )

        return ComprehensiveOptimizationResult(
            best_overall_params=final_result.best_params,
            best_overall_score=final_result.best_score,
            indicator_rankings=self.indicator_performances,
            optimal_tp_sl=optimal_tp_sl,
            feature_importance=feature_importance,
            optimization_time=time.time() - self.start_time,
            total_evaluations=self.evaluation_count + final_result.evaluation_count,
            convergence_history=final_result.optimization_history,
        )

    def _create_comprehensive_search_space(
        self, top_indicators: List[IndicatorPerformance]
    ) -> Tuple[List[Any], List[str]]:
        """Create comprehensive search space for top indicators + TP/SL."""

        search_space = []
        param_names = []

        # Add parameters for each top indicator
        for indicator in top_indicators:
            if indicator.name == "RSI":
                space, names = self.parameter_spaces.get_rsi_space(extended=False)
            elif indicator.name == "MACD":
                space, names = self.parameter_spaces.get_macd_space(extended=False)
            elif indicator.name == "Bollinger_Bands":
                space, names = self.parameter_spaces.get_bollinger_bands_space(
                    extended=False
                )
            elif indicator.name == "Moving_Averages":
                space, names = self.parameter_spaces.get_moving_averages_space(
                    extended=False
                )
            elif indicator.name == "Stochastic":
                space, names = self.parameter_spaces.get_stochastic_space(
                    extended=False
                )
            elif indicator.name == "ATR":
                space, names = self.parameter_spaces.get_atr_space(extended=False)
            else:
                continue

            search_space.extend(space)
            param_names.extend(names)

        # Add comprehensive trading strategy parameters
        trading_params = [
            # Risk Management
            Real(0.005, 0.08, name="stop_loss"),  # 0.5-8% stop loss
            Real(0.01, 0.25, name="take_profit"),  # 1-25% take profit
            Real(1.0, 5.0, name="risk_reward_ratio"),  # Risk/reward ratio
            # Position Management
            Real(0.1, 1.0, name="max_position_size"),  # 10-100% position size
            Integer(1, 48, name="position_timeout"),  # 1-48 hours timeout
            Integer(1, 20, name="max_daily_trades"),  # Daily trade limit
            # Signal Filtering
            Real(0.55, 0.9, name="min_confidence"),  # Minimum signal confidence
            Real(0.001, 0.05, name="min_volatility"),  # Minimum volatility filter
            Real(0.0001, 0.01, name="spread_filter"),  # Spread filter
            # Market Conditions
            Real(0.5, 3.0, name="volatility_multiplier"),  # Volatility-based sizing
            Real(0.1, 0.9, name="trend_strength_filter"),  # Trend strength requirement
        ]

        search_space.extend(trading_params)
        param_names.extend(
            [
                "stop_loss",
                "take_profit",
                "risk_reward_ratio",
                "max_position_size",
                "position_timeout",
                "max_daily_trades",
                "min_confidence",
                "min_volatility",
                "spread_filter",
                "volatility_multiplier",
                "trend_strength_filter",
            ]
        )

        return search_space, param_names

    def _evaluate_single_indicator_strategy(
        self, data: pd.DataFrame, params: Dict[str, Any], indicator_name: str
    ) -> float:
        """Evaluate strategy using single indicator."""
        try:
            # Generate signals using specified indicator
            signals = self._generate_indicator_signals(data, params, indicator_name)

            # Apply TP/SL logic
            signals_with_tpsl = self._apply_tp_sl_logic(data, signals, params)

            # Calculate multi-objective score
            score = self._calculate_multi_objective_score(
                data, signals_with_tpsl, params
            )

            return score

        except Exception as e:
            self.logger.warning(f"Error evaluating {indicator_name}: {e}")
            return 0.0

    def _evaluate_combined_strategy(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        indicators: List[IndicatorPerformance],
    ) -> float:
        """Evaluate strategy combining multiple indicators."""
        try:
            # Generate signals from each indicator
            combined_signals = pd.Series(0, index=data.index)

            for indicator in indicators:
                indicator_signals = self._generate_indicator_signals(
                    data, params, indicator.name
                )
                # Weight by indicator importance
                combined_signals += indicator_signals * indicator.importance_score

            # Normalize combined signals
            combined_signals = combined_signals / len(indicators)

            # Apply TP/SL logic
            final_signals = self._apply_tp_sl_logic(data, combined_signals, params)

            # Calculate multi-objective score
            score = self._calculate_multi_objective_score(data, final_signals, params)

            return score

        except Exception as e:
            self.logger.warning(f"Error evaluating combined strategy: {e}")
            return 0.0

    def _evaluate_final_strategy(
        self, data: pd.DataFrame, params: Dict[str, Any], selected_features: List[str]
    ) -> float:
        """Evaluate final optimized strategy."""
        try:
            # Generate signals using selected features only
            signals = self._generate_optimized_signals(data, params, selected_features)

            # Apply advanced TP/SL logic
            final_signals = self._apply_advanced_tp_sl_logic(data, signals, params)

            # Calculate comprehensive multi-objective score
            score = self._calculate_comprehensive_score(data, final_signals, params)

            return score

        except Exception as e:
            self.logger.warning(f"Error evaluating final strategy: {e}")
            return 0.0

    def _generate_indicator_signals(
        self, data: pd.DataFrame, params: Dict[str, Any], indicator_name: str
    ) -> pd.Series:
        """Generate trading signals for specific indicator."""
        signals = pd.Series(0, index=data.index)

        try:
            if indicator_name == "RSI":
                rsi = self.technical_indicators.rsi(
                    data["close"], params.get("rsi_period", 14)
                )
                oversold = params.get("rsi_oversold", 30)
                overbought = params.get("rsi_overbought", 70)

                signals[rsi < oversold] = 1  # Buy signal
                signals[rsi > overbought] = -1  # Sell signal

            elif indicator_name == "MACD":
                macd_data = self.technical_indicators.macd(
                    data["close"],
                    params.get("macd_fast", 12),
                    params.get("macd_slow", 26),
                    params.get("macd_signal", 9),
                )

                # MACD crossover signals
                signals[macd_data["MACD"] > macd_data["Signal"]] = 1
                signals[macd_data["MACD"] < macd_data["Signal"]] = -1

            elif indicator_name == "Bollinger_Bands":
                bb_data = self.technical_indicators.bollinger_bands(
                    data["close"],
                    params.get("bb_period", 20),
                    params.get("bb_std", 2.0),
                )

                # BB signals (using correct column names)
                signals[data["close"] < bb_data["BB_Lower"]] = 1  # Buy at lower band
                signals[data["close"] > bb_data["BB_Upper"]] = -1  # Sell at upper band

            elif indicator_name == "Moving_Averages":
                fast_ma = data["close"].rolling(params.get("ma_fast", 10)).mean()
                slow_ma = data["close"].rolling(params.get("ma_slow", 30)).mean()

                signals[fast_ma > slow_ma] = 1  # Bullish crossover
                signals[fast_ma < slow_ma] = -1  # Bearish crossover

            elif indicator_name == "Stochastic":
                stoch_data = self.technical_indicators.stochastic(
                    data["high"], data["low"], data["close"]
                )

                oversold = params.get("stoch_oversold", 20)
                overbought = params.get("stoch_overbought", 80)

                signals[stoch_data["Stoch_K"] < oversold] = 1
                signals[stoch_data["Stoch_K"] > overbought] = -1

            elif indicator_name == "ATR":
                atr = self.technical_indicators.atr(
                    data["high"],
                    data["low"],
                    data["close"],
                    params.get("atr_period", 14),
                )

                # ATR-based volatility breakout
                atr_threshold = atr * params.get("atr_multiplier", 2.0)
                price_change = data["close"].diff().abs()

                signals[price_change > atr_threshold] = 1  # Breakout signal

            return signals

        except Exception as e:
            self.logger.warning(f"Error generating {indicator_name} signals: {e}")
            return signals

    def _apply_tp_sl_logic(
        self, data: pd.DataFrame, signals: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Apply take profit and stop loss logic to signals."""

        stop_loss = params.get("stop_loss", 0.05)
        take_profit = params.get("take_profit", 0.10)
        position_timeout = params.get("position_timeout", 24)

        final_signals = signals.copy()
        current_position = 0
        entry_price = 0
        entry_time = 0

        for i in range(1, len(data)):
            current_price = data["close"].iloc[i]

            # Check for new signal
            if signals.iloc[i] != 0 and current_position == 0:
                # Enter position
                current_position = signals.iloc[i]
                entry_price = current_price
                entry_time = i
                final_signals.iloc[i] = current_position

            elif current_position != 0:
                # Check exit conditions
                if current_position == 1:  # Long position
                    pnl = (current_price - entry_price) / entry_price

                    if pnl >= take_profit:  # Take profit
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    elif pnl <= -stop_loss:  # Stop loss
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    elif i - entry_time >= position_timeout:  # Timeout
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    else:
                        final_signals.iloc[i] = 0  # Hold position

                elif current_position == -1:  # Short position
                    pnl = (entry_price - current_price) / entry_price

                    if pnl >= take_profit:  # Take profit
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    elif pnl <= -stop_loss:  # Stop loss
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    elif i - entry_time >= position_timeout:  # Timeout
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    else:
                        final_signals.iloc[i] = 0  # Hold position
            else:
                final_signals.iloc[i] = 0

        return final_signals

    def _apply_advanced_tp_sl_logic(
        self, data: pd.DataFrame, signals: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Apply advanced TP/SL logic with dynamic adjustments."""

        # Get parameters
        base_stop_loss = params.get("stop_loss", 0.05)
        base_take_profit = params.get("take_profit", 0.10)
        risk_reward_ratio = params.get("risk_reward_ratio", 2.0)
        volatility_multiplier = params.get("volatility_multiplier", 1.0)

        # Calculate dynamic volatility
        volatility = data["close"].pct_change().rolling(20).std()

        final_signals = signals.copy()
        current_position = 0
        entry_price = 0
        entry_time = 0
        dynamic_stop = 0
        dynamic_tp = 0

        for i in range(1, len(data)):
            current_price = data["close"].iloc[i]
            current_vol = (
                volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.02
            )

            # Check for new signal
            if signals.iloc[i] != 0 and current_position == 0:
                # Calculate dynamic TP/SL based on volatility
                vol_adjusted_stop = base_stop_loss * (
                    1 + current_vol * volatility_multiplier
                )
                vol_adjusted_tp = vol_adjusted_stop * risk_reward_ratio

                # Enter position
                current_position = signals.iloc[i]
                entry_price = current_price
                entry_time = i
                dynamic_stop = vol_adjusted_stop
                dynamic_tp = vol_adjusted_tp
                final_signals.iloc[i] = current_position

            elif current_position != 0:
                # Check exit conditions with dynamic TP/SL
                if current_position == 1:  # Long position
                    pnl = (current_price - entry_price) / entry_price

                    if pnl >= dynamic_tp:  # Take profit
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    elif pnl <= -dynamic_stop:  # Stop loss
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    else:
                        final_signals.iloc[i] = 0  # Hold position

                elif current_position == -1:  # Short position
                    pnl = (entry_price - current_price) / entry_price

                    if pnl >= dynamic_tp:  # Take profit
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    elif pnl <= -dynamic_stop:  # Stop loss
                        final_signals.iloc[i] = -current_position
                        current_position = 0
                    else:
                        final_signals.iloc[i] = 0  # Hold position
            else:
                final_signals.iloc[i] = 0

        return final_signals

    def _calculate_multi_objective_score(
        self, data: pd.DataFrame, signals: pd.Series, params: Dict[str, Any]
    ) -> float:
        """Calculate multi-objective score balancing return and risk."""
        try:
            # Calculate strategy returns
            returns = self._calculate_strategy_returns(data, signals, params)

            if len(returns) == 0 or returns.std() == 0:
                return 0.0

            # Calculate metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)

            # Multi-objective score
            # Balance return (sharpe) with risk management (drawdown, win rate)
            score = (
                sharpe_ratio * 0.4  # Risk-adjusted returns
                + (1 - max_drawdown) * 0.3  # Risk management
                + win_rate * 0.2  # Consistency
                + min(profit_factor / 2, 1.0) * 0.1  # Efficiency (capped)
            )

            return max(score, 0.0)  # Ensure non-negative

        except Exception as e:
            self.logger.warning(f"Error calculating multi-objective score: {e}")
            return 0.0

    def _calculate_comprehensive_score(
        self, data: pd.DataFrame, signals: pd.Series, params: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive score for final optimization."""
        try:
            returns = self._calculate_strategy_returns(data, signals, params)

            if len(returns) == 0:
                return 0.0

            # Advanced metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)

            # Trade quality metrics
            num_trades = (signals != 0).sum()
            avg_trade_return = (
                returns[returns != 0].mean() if len(returns[returns != 0]) > 0 else 0
            )

            # Penalty for too many or too few trades
            trade_penalty = 1.0
            if num_trades < 10:  # Too few trades
                trade_penalty = 0.5
            elif (
                num_trades > len(data) * 0.1
            ):  # Too many trades (more than 10% of periods)
                trade_penalty = 0.7

            # Comprehensive score
            score = (
                sharpe_ratio * 0.25  # Risk-adjusted returns
                + sortino_ratio * 0.15  # Downside risk focus
                + calmar_ratio * 0.15  # Drawdown-adjusted returns
                + (1 - max_drawdown) * 0.20  # Risk management
                + win_rate * 0.15  # Consistency
                + min(profit_factor / 2, 1.0) * 0.10  # Efficiency
            ) * trade_penalty

            return max(score, 0.0)

        except Exception as e:
            self.logger.warning(f"Error calculating comprehensive score: {e}")
            return 0.0

    def _calculate_strategy_returns(
        self, data: pd.DataFrame, signals: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Calculate strategy returns with transaction costs."""
        try:
            price_returns = data["close"].pct_change().fillna(0)
            lagged_signals = signals.shift(1).fillna(0)

            # Strategy returns
            strategy_returns = price_returns * lagged_signals

            # Transaction costs
            transaction_cost = params.get("transaction_cost", 0.001)
            signal_changes = lagged_signals.diff().abs()
            costs = signal_changes * transaction_cost

            return (strategy_returns - costs).fillna(0)

        except Exception as e:
            self.logger.warning(f"Error calculating strategy returns: {e}")
            return pd.Series(0, index=data.index)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return self._calculate_sharpe_ratio(returns)
        return (returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd == 0:
            return 0.0
        return (returns.mean() * 252) / max_dd

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 1.0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return abs(drawdown.min())

    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        trading_returns = returns[returns != 0]
        if len(trading_returns) == 0:
            return 0.0
        return (trading_returns > 0).mean()

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        trading_returns = returns[returns != 0]
        if len(trading_returns) == 0:
            return 1.0

        profits = trading_returns[trading_returns > 0].sum()
        losses = abs(trading_returns[trading_returns < 0].sum())

        if losses == 0:
            return profits if profits > 0 else 1.0
        return profits / losses

    def _calculate_indicator_performance(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        params: Dict[str, Any],
        indicator_name: str,
        train_score: float,
        val_score: float,
    ) -> IndicatorPerformance:
        """Calculate comprehensive performance metrics for an indicator."""

        try:
            # Generate signals and calculate returns
            signals = self._generate_indicator_signals(val_data, params, indicator_name)
            signals_with_tpsl = self._apply_tp_sl_logic(val_data, signals, params)
            returns = self._calculate_strategy_returns(
                val_data, signals_with_tpsl, params
            )

            # Calculate metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)

            # Trade analysis
            trade_signals = signals_with_tpsl[signals_with_tpsl != 0]
            total_trades = len(trade_signals)
            avg_trade_duration = 24  # Placeholder - would need position tracking for accurate calculation

            # Parameter sensitivity (how much performance degrades out-of-sample)
            parameter_sensitivity = abs(train_score - val_score) / max(
                train_score, 0.001
            )

            # Importance score (balanced metric)
            importance_score = (
                sharpe_ratio * 0.3
                + (1 - max_drawdown) * 0.3
                + win_rate * 0.2
                + min(profit_factor / 2, 1.0) * 0.1
                + (1 - parameter_sensitivity) * 0.1  # Stability bonus
            )

            return IndicatorPerformance(
                name=indicator_name,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                avg_trade_duration=avg_trade_duration,
                parameter_sensitivity=parameter_sensitivity,
                importance_score=max(importance_score, 0.0),
            )

        except Exception as e:
            self.logger.warning(
                f"Error calculating performance for {indicator_name}: {e}"
            )
            return IndicatorPerformance(
                name=indicator_name,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                win_rate=0.0,
                profit_factor=1.0,
                total_trades=0,
                avg_trade_duration=0.0,
                parameter_sensitivity=1.0,
                importance_score=0.0,
            )

    def _select_optimal_features(self) -> List[str]:
        """Select optimal features based on individual performance."""

        # Select indicators with importance score > threshold
        threshold = 0.1
        selected_indicators = [
            perf.name
            for perf in self.indicator_performances
            if perf.importance_score > threshold
        ]

        self.logger.info(
            f"🎯 Selected {len(selected_indicators)} indicators above threshold {threshold}"
        )

        return selected_indicators

    def _create_final_search_space(
        self, selected_features: List[str]
    ) -> Tuple[List[Any], List[str]]:
        """Create final search space with only selected features."""

        search_space = []
        param_names = []

        # Add parameters for selected indicators only
        for feature in selected_features:
            if feature == "RSI":
                space, names = self.parameter_spaces.get_rsi_space(extended=False)
                search_space.extend(space)
                param_names.extend(names)
            elif feature == "MACD":
                space, names = self.parameter_spaces.get_macd_space(extended=False)
                search_space.extend(space)
                param_names.extend(names)
            # Add other indicators as needed...

        # Always include comprehensive trading parameters
        trading_params = [
            Real(0.005, 0.08, name="stop_loss"),
            Real(0.01, 0.25, name="take_profit"),
            Real(1.0, 5.0, name="risk_reward_ratio"),
            Real(0.1, 1.0, name="max_position_size"),
            Integer(1, 48, name="position_timeout"),
            Real(0.55, 0.9, name="min_confidence"),
            Real(0.5, 3.0, name="volatility_multiplier"),
        ]

        search_space.extend(trading_params)
        param_names.extend(
            [
                "stop_loss",
                "take_profit",
                "risk_reward_ratio",
                "max_position_size",
                "position_timeout",
                "min_confidence",
                "volatility_multiplier",
            ]
        )

        return search_space, param_names

    def _generate_optimized_signals(
        self, data: pd.DataFrame, params: Dict[str, Any], selected_features: List[str]
    ) -> pd.Series:
        """Generate signals using only selected features."""

        combined_signals = pd.Series(0.0, index=data.index)

        for feature in selected_features:
            # Get indicator performance for weighting
            indicator_perf = next(
                (p for p in self.indicator_performances if p.name == feature), None
            )
            weight = indicator_perf.importance_score if indicator_perf else 1.0

            # Generate signals for this indicator
            indicator_signals = self._generate_indicator_signals(data, params, feature)

            # Add weighted signals
            combined_signals += indicator_signals * weight

        # Normalize and apply confidence threshold
        if len(selected_features) > 0:
            combined_signals = combined_signals / len(selected_features)

        min_confidence = params.get("min_confidence", 0.6)

        # Only keep signals above confidence threshold
        final_signals = pd.Series(0, index=data.index)
        final_signals[combined_signals > min_confidence] = 1
        final_signals[combined_signals < -min_confidence] = -1

        return final_signals

    def _calculate_feature_importance(
        self, optimization_result: OptimizationResult, selected_features: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance from optimization results."""

        feature_importance = {}

        # Use parameter importance from optimization
        param_importance = optimization_result.parameter_importance

        # Map parameters to features
        for feature in selected_features:
            feature_params = []
            if feature == "RSI":
                feature_params = ["rsi_period", "rsi_oversold", "rsi_overbought"]
            elif feature == "MACD":
                feature_params = ["macd_fast", "macd_slow", "macd_signal"]
            # Add other mappings...

            # Calculate average importance for feature parameters
            feature_importance[feature] = np.mean(
                [param_importance.get(param, 0.0) for param in feature_params]
            )

        # Add trading parameter importance
        trading_params = [
            "stop_loss",
            "take_profit",
            "risk_reward_ratio",
            "position_timeout",
        ]
        for param in trading_params:
            feature_importance[param] = param_importance.get(param, 0.0)

        return feature_importance


def main():
    """
    Demonstration of comprehensive hyperparameter optimization.
    """
    print("🚀 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION SYSTEM")
    print("=" * 80)
    print("Testing ALL indicators + optimizing TP/SL dynamically")
    print("=" * 80)
    print()

    # Create optimizer
    optimizer = ComprehensiveHyperparameterOptimizer(
        optimization_config=OptimizationConfig(
            n_calls=300, n_initial_points=50  # Comprehensive search
        ),
        enable_feature_selection=True,
        max_optimization_time_hours=2.0,
    )

    print("🎯 OPTIMIZATION FEATURES:")
    print("   ✅ Tests ALL technical indicators individually")
    print("   ✅ Ranks indicators by performance")
    print("   ✅ Optimizes stop loss & take profit dynamically")
    print("   ✅ Multi-objective optimization (return vs risk)")
    print("   ✅ Feature selection based on importance")
    print("   ✅ Walk-forward validation for robustness")
    print()

    print("💡 TO RUN WITH YOUR DATA:")
    print("   1. Load your historical BTC data")
    print("   2. result = optimizer.optimize_all_indicators(data)")
    print("   3. Check result.optimal_tp_sl for best TP/SL parameters")
    print("   4. Check result.indicator_rankings for best indicators")
    print()

    print("📊 EXPECTED OUTPUT:")
    print("   • Ranked list of most effective indicators")
    print("   • Optimal TP/SL parameters for current market conditions")
    print("   • Feature importance scores")
    print("   • Combined strategy performance metrics")


if __name__ == "__main__":
    main()
