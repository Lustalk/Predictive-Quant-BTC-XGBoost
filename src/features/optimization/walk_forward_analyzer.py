"""
Walk-Forward Analysis for Parameter Robustness Testing
====================================================

Implements walk-forward optimization to ensure parameter stability across
different time periods and market conditions. This prevents overfitting
and validates parameter robustness in out-of-sample data.

Features:
- Rolling window optimization
- Out-of-sample validation
- Parameter stability analysis
- Performance degradation detection
- Market regime awareness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .smart_optimizer import SmartOptimizer, OptimizationConfig
from .objective_functions import ObjectiveFunctions
from ...utils.logging import get_logger


@dataclass
class WalkForwardPeriod:
    """Single walk-forward period results"""
    period_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any]
    in_sample_score: float
    out_of_sample_score: float
    performance_degradation: float
    num_trades: int


@dataclass
class WalkForwardResults:
    """Complete walk-forward analysis results"""
    periods: List[WalkForwardPeriod]
    overall_stability: float
    parameter_consistency: Dict[str, float]
    average_degradation: float
    best_stable_params: Dict[str, Any]
    performance_summary: Dict[str, float]


class WalkForwardAnalyzer:
    """
    Walk-forward analyzer for parameter robustness testing.
    
    Implements rolling window optimization where parameters are optimized
    on training data and validated on unseen test data.
    """
    
    def __init__(self, 
                 optimizer_config: OptimizationConfig = None,
                 min_train_days: int = 90,
                 test_days: int = 30,
                 step_days: int = 15):
        """
        Initialize walk-forward analyzer.
        
        Args:
            optimizer_config: Configuration for Bayesian optimizer
            min_train_days: Minimum training period in days
            test_days: Testing period in days
            step_days: Step size between periods in days
        """
        self.optimizer_config = optimizer_config or OptimizationConfig(n_calls=50)  # Fewer calls per period
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.step_days = step_days
        
        self.logger = get_logger().get_logger()
        self.objective_functions = ObjectiveFunctions()
    
    def run_walk_forward_analysis(self,
                                data: pd.DataFrame,
                                parameter_space: List[Any],
                                objective_function: Callable,
                                parameter_names: List[str],
                                max_periods: int = 10) -> WalkForwardResults:
        """
        Run complete walk-forward analysis.
        
        Args:
            data: Historical OHLCV data with datetime index
            parameter_space: Search space for optimization
            objective_function: Objective function to optimize
            parameter_names: Names of parameters being optimized
            max_periods: Maximum number of walk-forward periods
            
        Returns:
            WalkForwardResults with complete analysis
        """
        self.logger.info(f"Starting walk-forward analysis with {max_periods} periods")
        
        # Generate walk-forward periods
        periods = self._generate_periods(data, max_periods)
        
        if len(periods) == 0:
            raise ValueError("No valid walk-forward periods could be generated")
        
        self.logger.info(f"Generated {len(periods)} walk-forward periods")
        
        # Run optimization for each period
        results = []
        for i, (train_data, test_data, train_start, train_end, test_start, test_end) in enumerate(periods):
            self.logger.info(f"Processing period {i+1}/{len(periods)}: "
                           f"Train: {train_start.date()} to {train_end.date()}, "
                           f"Test: {test_start.date()} to {test_end.date()}")
            
            try:
                # Optimize on training data
                optimizer = SmartOptimizer(self.optimizer_config)
                optimization_result = optimizer.optimize_indicators(
                    data=train_data,
                    parameter_space=parameter_space,
                    objective_function=objective_function,
                    parameter_names=parameter_names
                )
                
                # Validate on test data
                test_score = objective_function(test_data, optimization_result.best_params)
                
                # Calculate performance degradation
                degradation = self._calculate_degradation(
                    optimization_result.best_score, test_score
                )
                
                # Count trades in test period
                num_trades = self._count_trades(test_data, optimization_result.best_params)
                
                # Create period result
                period_result = WalkForwardPeriod(
                    period_id=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    best_params=optimization_result.best_params,
                    in_sample_score=optimization_result.best_score,
                    out_of_sample_score=test_score,
                    performance_degradation=degradation,
                    num_trades=num_trades
                )
                
                results.append(period_result)
                
                self.logger.info(f"Period {i+1} completed: "
                                f"In-sample: {optimization_result.best_score:.4f}, "
                                f"Out-of-sample: {test_score:.4f}, "
                                f"Degradation: {degradation:.2%}")
                
            except Exception as e:
                self.logger.error(f"Error processing period {i+1}: {e}")
                continue
        
        if len(results) == 0:
            raise ValueError("No walk-forward periods completed successfully")
        
        # Analyze results
        analysis_results = self._analyze_walk_forward_results(results, parameter_names)
        
        self.logger.info(f"Walk-forward analysis completed:")
        self.logger.info(f"Average degradation: {analysis_results.average_degradation:.2%}")
        self.logger.info(f"Overall stability: {analysis_results.overall_stability:.4f}")
        
        return analysis_results
    
    def analyze_parameter_stability(self, 
                                  results: WalkForwardResults) -> Dict[str, Any]:
        """
        Analyze parameter stability across walk-forward periods.
        
        Args:
            results: Walk-forward analysis results
            
        Returns:
            Parameter stability analysis
        """
        if len(results.periods) == 0:
            return {}
        
        # Extract parameter values across periods
        all_params = {}
        for period in results.periods:
            for param_name, param_value in period.best_params.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(param_value)
        
        # Calculate stability metrics for each parameter
        stability_analysis = {}
        
        for param_name, values in all_params.items():
            values = np.array(values)
            
            # Coefficient of variation (stability measure)
            if values.std() > 0:
                cv = values.std() / np.abs(values.mean()) if values.mean() != 0 else float('inf')
                stability = 1.0 / (1.0 + cv)  # Higher stability = lower variation
            else:
                stability = 1.0
            
            # Trend analysis
            if len(values) > 2:
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                trend_strength = abs(trend_slope) / values.std() if values.std() > 0 else 0
            else:
                trend_slope = 0
                trend_strength = 0
            
            stability_analysis[param_name] = {
                'values': values.tolist(),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'stability_score': float(stability),
                'coefficient_of_variation': float(cv) if cv != float('inf') else None,
                'trend_slope': float(trend_slope),
                'trend_strength': float(trend_strength)
            }
        
        return stability_analysis
    
    def find_robust_parameters(self,
                             results: WalkForwardResults,
                             stability_threshold: float = 0.7,
                             min_performance: float = 0.0) -> Dict[str, Any]:
        """
        Find robust parameters that perform consistently across periods.
        
        Args:
            results: Walk-forward analysis results
            stability_threshold: Minimum stability score
            min_performance: Minimum average out-of-sample performance
            
        Returns:
            Robust parameter recommendations
        """
        if len(results.periods) == 0:
            return {}
        
        # Filter periods by minimum performance
        good_periods = [
            period for period in results.periods
            if period.out_of_sample_score >= min_performance
        ]
        
        if len(good_periods) == 0:
            self.logger.warning("No periods meet minimum performance criteria")
            good_periods = results.periods
        
        # Calculate parameter statistics from good periods
        param_stats = {}
        all_params = set()
        
        for period in good_periods:
            all_params.update(period.best_params.keys())
        
        for param_name in all_params:
            values = [
                period.best_params[param_name] 
                for period in good_periods 
                if param_name in period.best_params
            ]
            
            if len(values) > 0:
                values = np.array(values)
                param_stats[param_name] = {
                    'median': float(np.median(values)),
                    'mean': float(np.mean(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                    'stability': results.parameter_consistency.get(param_name, 0.0)
                }
        
        # Select robust parameters (use median for stability)
        robust_params = {}
        for param_name, stats in param_stats.items():
            if stats['stability'] >= stability_threshold:
                robust_params[param_name] = stats['median']
            else:
                # Use median even if not fully stable
                robust_params[param_name] = stats['median']
                self.logger.warning(f"Parameter {param_name} has low stability: {stats['stability']:.3f}")
        
        return {
            'robust_parameters': robust_params,
            'parameter_statistics': param_stats,
            'periods_used': len(good_periods),
            'stability_threshold': stability_threshold,
            'min_performance': min_performance
        }
    
    def _generate_periods(self, 
                         data: pd.DataFrame, 
                         max_periods: int) -> List[Tuple]:
        """Generate walk-forward periods from data"""
        periods = []
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date
        period_count = 0
        
        while period_count < max_periods:
            # Calculate period boundaries
            train_start = current_date
            train_end = train_start + timedelta(days=self.min_train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_days)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            # Extract period data
            train_mask = (data.index >= train_start) & (data.index <= train_end)
            test_mask = (data.index >= test_start) & (data.index <= test_end)
            
            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()
            
            # Ensure we have enough data points
            if len(train_data) < 50 or len(test_data) < 10:
                current_date += timedelta(days=self.step_days)
                continue
            
            periods.append((
                train_data, test_data, 
                train_start, train_end,
                test_start, test_end
            ))
            
            period_count += 1
            current_date += timedelta(days=self.step_days)
        
        return periods
    
    def _calculate_degradation(self, in_sample_score: float, out_of_sample_score: float) -> float:
        """Calculate performance degradation"""
        if in_sample_score <= 0:
            return 1.0  # Maximum degradation
        
        degradation = (in_sample_score - out_of_sample_score) / in_sample_score
        return max(0.0, degradation)  # Prevent negative degradation
    
    def _count_trades(self, data: pd.DataFrame, params: Dict[str, Any]) -> int:
        """Count number of trades in test period"""
        try:
            # Use the objective function's signal generation
            signals = self.objective_functions._generate_signals(data, params)
            
            # Count signal changes (trades)
            signal_changes = signals.diff().abs()
            num_trades = (signal_changes > 0).sum()
            
            return int(num_trades)
            
        except Exception as e:
            self.logger.warning(f"Error counting trades: {e}")
            return 0
    
    def _analyze_walk_forward_results(self, 
                                    periods: List[WalkForwardPeriod],
                                    parameter_names: List[str]) -> WalkForwardResults:
        """Analyze walk-forward period results"""
        
        # Calculate overall metrics
        in_sample_scores = [p.in_sample_score for p in periods]
        out_of_sample_scores = [p.out_of_sample_score for p in periods]
        degradations = [p.performance_degradation for p in periods]
        
        # Overall stability (correlation between in-sample and out-of-sample)
        if len(in_sample_scores) > 2:
            overall_stability = np.corrcoef(in_sample_scores, out_of_sample_scores)[0, 1]
            if np.isnan(overall_stability):
                overall_stability = 0.0
        else:
            overall_stability = 0.0
        
        # Parameter consistency analysis
        parameter_consistency = {}
        for param_name in parameter_names:
            param_values = []
            for period in periods:
                if param_name in period.best_params:
                    param_values.append(period.best_params[param_name])
            
            if len(param_values) > 1:
                # Coefficient of variation (lower is more consistent)
                cv = np.std(param_values) / np.abs(np.mean(param_values)) if np.mean(param_values) != 0 else float('inf')
                consistency = 1.0 / (1.0 + cv) if cv != float('inf') else 1.0
                parameter_consistency[param_name] = consistency
            else:
                parameter_consistency[param_name] = 1.0
        
        # Find best stable parameters (highest consistency-weighted performance)
        best_stable_params = {}
        if periods:
            # Weight each period's parameters by their out-of-sample performance and consistency
            weighted_params = {}
            total_weight = 0
            
            for period in periods:
                weight = max(0, period.out_of_sample_score)  # Use out-of-sample score as weight
                total_weight += weight
                
                for param_name, param_value in period.best_params.items():
                    if param_name not in weighted_params:
                        weighted_params[param_name] = 0
                    weighted_params[param_name] += param_value * weight
            
            # Calculate weighted average
            if total_weight > 0:
                for param_name, weighted_sum in weighted_params.items():
                    best_stable_params[param_name] = weighted_sum / total_weight
        
        # Performance summary
        performance_summary = {
            'average_in_sample': np.mean(in_sample_scores),
            'average_out_of_sample': np.mean(out_of_sample_scores),
            'std_in_sample': np.std(in_sample_scores),
            'std_out_of_sample': np.std(out_of_sample_scores),
            'best_out_of_sample': max(out_of_sample_scores) if out_of_sample_scores else 0,
            'worst_out_of_sample': min(out_of_sample_scores) if out_of_sample_scores else 0,
            'positive_periods': sum(1 for score in out_of_sample_scores if score > 0),
            'total_periods': len(periods)
        }
        
        return WalkForwardResults(
            periods=periods,
            overall_stability=overall_stability,
            parameter_consistency=parameter_consistency,
            average_degradation=np.mean(degradations),
            best_stable_params=best_stable_params,
            performance_summary=performance_summary
        )
    
    def save_walk_forward_report(self, 
                               results: WalkForwardResults,
                               save_path: str = "walk_forward_analysis") -> None:
        """Save comprehensive walk-forward analysis report"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: In-sample vs Out-of-sample performance
            periods = [p.period_id for p in results.periods]
            in_sample = [p.in_sample_score for p in results.periods]
            out_of_sample = [p.out_of_sample_score for p in results.periods]
            
            axes[0, 0].plot(periods, in_sample, 'b-o', label='In-sample', alpha=0.7)
            axes[0, 0].plot(periods, out_of_sample, 'r-o', label='Out-of-sample', alpha=0.7)
            axes[0, 0].set_title('Performance Across Walk-Forward Periods')
            axes[0, 0].set_xlabel('Period')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Performance degradation
            degradation = [p.performance_degradation for p in results.periods]
            axes[0, 1].bar(periods, degradation, alpha=0.7, color='orange')
            axes[0, 1].set_title('Performance Degradation by Period')
            axes[0, 1].set_xlabel('Period')
            axes[0, 1].set_ylabel('Degradation')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Parameter consistency
            param_names = list(results.parameter_consistency.keys())[:10]  # Top 10 parameters
            param_consistency = [results.parameter_consistency[name] for name in param_names]
            
            axes[1, 0].barh(param_names, param_consistency, alpha=0.7, color='green')
            axes[1, 0].set_title('Parameter Consistency')
            axes[1, 0].set_xlabel('Consistency Score')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Stability metrics
            metrics = ['Overall Stability', 'Avg Degradation', 'Positive Periods %']
            values = [
                results.overall_stability,
                -results.average_degradation,  # Negative for visualization
                results.performance_summary['positive_periods'] / results.performance_summary['total_periods']
            ]
            
            colors = ['blue', 'red', 'green']
            axes[1, 1].bar(metrics, values, alpha=0.7, color=colors)
            axes[1, 1].set_title('Summary Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}_report.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Walk-forward analysis report saved to {save_path}_report.png")
            
        except Exception as e:
            self.logger.warning(f"Could not save walk-forward report: {e}")