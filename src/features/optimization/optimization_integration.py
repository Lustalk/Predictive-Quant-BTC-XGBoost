#!/usr/bin/env python3
"""
🔗 Optimization Integration Module
=================================

Integrates comprehensive hyperparameter optimization with the existing system.
Provides easy-to-use interfaces for:
1. Running comprehensive optimization
2. Applying optimized parameters
3. Updating configuration dynamically
"""

import json
import yaml
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from .comprehensive_hyperparameter_optimizer import (
    ComprehensiveHyperparameterOptimizer, 
    OptimizationConfig,
    ComprehensiveOptimizationResult
)
from ...utils.logging import get_logger


class OptimizationIntegration:
    """
    Integration layer for comprehensive optimization with existing system
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize optimization integration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.logger = get_logger().get_logger()
        self.results_path = Path("comprehensive_optimization_results.json")
        
    def run_comprehensive_optimization(self, 
                                     data: pd.DataFrame,
                                     save_results: bool = True,
                                     update_config: bool = True) -> ComprehensiveOptimizationResult:
        """
        Run comprehensive optimization and optionally update configuration
        
        Args:
            data: Historical OHLCV data
            save_results: Whether to save results to file
            update_config: Whether to update config file with optimal parameters
            
        Returns:
            ComprehensiveOptimizationResult
        """
        self.logger.info("🚀 Starting comprehensive optimization integration...")
        
        # Create optimizer with production settings
        optimizer = ComprehensiveHyperparameterOptimizer(
            optimization_config=OptimizationConfig(
                n_calls=300,  # Comprehensive search
                n_initial_points=50,
                acquisition_function='EI'
            ),
            enable_feature_selection=True,
            max_optimization_time_hours=3.0  # 3 hours for production
        )
        
        # Run optimization
        result = optimizer.optimize_all_indicators(
            data=data,
            validation_split=0.2
        )
        
        self.logger.info(f"✅ Optimization complete! Best score: {result.best_overall_score:.3f}")
        
        # Save results if requested
        if save_results:
            self._save_optimization_results(result)
        
        # Update configuration if requested
        if update_config:
            self._update_configuration_with_results(result)
        
        return result
    
    def load_optimization_results(self) -> Optional[Dict[str, Any]]:
        """
        Load previous optimization results from file
        
        Returns:
            Dictionary with optimization results or None if not found
        """
        try:
            if self.results_path.exists():
                with open(self.results_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning("No previous optimization results found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading optimization results: {e}")
            return None
    
    def apply_optimized_parameters(self, 
                                 optimization_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply optimized parameters to current configuration
        
        Args:
            optimization_results: Results dict (if None, loads from file)
            
        Returns:
            Updated configuration dictionary
        """
        # Load results if not provided
        if optimization_results is None:
            optimization_results = self.load_optimization_results()
            
        if optimization_results is None:
            self.logger.error("No optimization results available")
            return {}
        
        # Load current configuration
        current_config = self._load_current_configuration()
        
        # Apply optimized TP/SL parameters
        optimal_tp_sl = optimization_results.get('optimal_tp_sl', {})
        
        if optimal_tp_sl:
            current_config['strategy']['risk_management']['stop_loss'] = optimal_tp_sl.get('stop_loss', 0.05)
            current_config['strategy']['risk_management']['take_profit'] = optimal_tp_sl.get('take_profit', 0.10)
            
            # Add new optimized parameters
            current_config['strategy']['risk_management']['risk_reward_ratio'] = optimal_tp_sl.get('risk_reward_ratio', 2.0)
            current_config['strategy']['risk_management']['position_timeout_hours'] = optimal_tp_sl.get('position_timeout', 24)
            
            self.logger.info("✅ Applied optimized TP/SL parameters to configuration")
        
        # Update indicator priorities based on rankings
        top_indicators = optimization_results.get('top_indicators', [])
        if top_indicators:
            current_config['features']['indicator_priorities'] = top_indicators
            self.logger.info(f"✅ Updated indicator priorities: {top_indicators}")
        
        return current_config
    
    def create_optimized_strategy_config(self, 
                                       optimization_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a complete strategy configuration based on optimization results
        
        Args:
            optimization_results: Results dict (if None, loads from file)
            
        Returns:
            Complete optimized strategy configuration
        """
        # Load results if not provided
        if optimization_results is None:
            optimization_results = self.load_optimization_results()
            
        if optimization_results is None:
            self.logger.error("No optimization results available")
            return {}
        
        # Get optimal parameters
        optimal_tp_sl = optimization_results.get('optimal_tp_sl', {})
        top_indicators = optimization_results.get('top_indicators', [])
        indicator_scores = optimization_results.get('indicator_scores', {})
        
        # Create optimized configuration
        optimized_config = {
            'strategy_name': 'Optimized BTC Strategy',
            'optimization_timestamp': optimization_results.get('timestamp'),
            'optimization_score': optimization_results.get('best_score', 0.0),
            
            # Risk Management (optimized)
            'risk_management': {
                'stop_loss': optimal_tp_sl.get('stop_loss', 0.05),
                'take_profit': optimal_tp_sl.get('take_profit', 0.10),
                'risk_reward_ratio': optimal_tp_sl.get('risk_reward_ratio', 2.0),
                'position_timeout_hours': optimal_tp_sl.get('position_timeout', 24),
                'max_drawdown': 0.15,  # Conservative
                'max_daily_trades': 10
            },
            
            # Indicator Configuration (optimized)
            'indicators': {
                'enabled_indicators': top_indicators[:5],  # Top 5 indicators
                'indicator_weights': {
                    indicator: score for indicator, score in indicator_scores.items()
                },
                'confidence_threshold': 0.6  # Minimum confidence for signals
            },
            
            # Position Sizing (optimized)
            'position_sizing': {
                'method': 'volatility_adjusted',
                'base_position_size': 0.25,  # 25% of capital
                'max_position_size': 1.0,    # 100% max
                'volatility_lookback': 20    # Days for volatility calculation
            },
            
            # Signal Generation
            'signal_generation': {
                'min_confidence': 0.6,
                'trend_filter': True,
                'volatility_filter': True,
                'correlation_filter': True
            }
        }
        
        return optimized_config
    
    def _save_optimization_results(self, result: ComprehensiveOptimizationResult):
        """Save optimization results to JSON file"""
        try:
            results_dict = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'best_score': result.best_overall_score,
                'optimization_time': result.optimization_time,
                'total_evaluations': result.total_evaluations,
                'optimal_tp_sl': result.optimal_tp_sl,
                'top_indicators': [ind.name for ind in result.indicator_rankings[:10]],
                'indicator_scores': {
                    ind.name: ind.importance_score 
                    for ind in result.indicator_rankings
                },
                'indicator_details': {
                    ind.name: {
                        'sharpe_ratio': ind.sharpe_ratio,
                        'max_drawdown': ind.max_drawdown,
                        'win_rate': ind.win_rate,
                        'profit_factor': ind.profit_factor,
                        'importance_score': ind.importance_score
                    }
                    for ind in result.indicator_rankings
                },
                'feature_importance': result.feature_importance,
                'best_overall_params': result.best_overall_params
            }
            
            with open(self.results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            self.logger.info(f"✅ Optimization results saved to {self.results_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")
    
    def _update_configuration_with_results(self, result: ComprehensiveOptimizationResult):
        """Update configuration file with optimization results"""
        try:
            # Load current configuration
            current_config = self._load_current_configuration()
            
            # Update with optimized parameters
            if 'strategy' not in current_config:
                current_config['strategy'] = {}
            if 'risk_management' not in current_config['strategy']:
                current_config['strategy']['risk_management'] = {}
            
            # Update TP/SL parameters
            risk_mgmt = current_config['strategy']['risk_management']
            tp_sl = result.optimal_tp_sl
            
            risk_mgmt['stop_loss'] = tp_sl.get('stop_loss', 0.05)
            risk_mgmt['take_profit'] = tp_sl.get('take_profit', 0.10)
            risk_mgmt['risk_reward_ratio'] = tp_sl.get('risk_reward_ratio', 2.0)
            risk_mgmt['position_timeout_hours'] = tp_sl.get('position_timeout', 24)
            
            # Add optimization metadata
            current_config['optimization'] = {
                'last_optimization': pd.Timestamp.now().isoformat(),
                'optimization_score': result.best_overall_score,
                'optimization_time_seconds': result.optimization_time,
                'total_evaluations': result.total_evaluations
            }
            
            # Add indicator priorities
            if 'features' not in current_config:
                current_config['features'] = {}
            
            current_config['features']['indicator_priorities'] = [
                ind.name for ind in result.indicator_rankings[:5]
            ]
            
            # Save updated configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(current_config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"✅ Configuration updated with optimization results")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
    
    def _load_current_configuration(self) -> Dict[str, Any]:
        """Load current configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}
    
    def get_optimization_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of the latest optimization results"""
        
        results = self.load_optimization_results()
        if not results:
            return None
        
        # Extract key information
        summary = {
            'optimization_date': results.get('timestamp', 'Unknown'),
            'performance_score': results.get('best_score', 0.0),
            'optimization_time_minutes': results.get('optimization_time', 0) / 60,
            'total_evaluations': results.get('total_evaluations', 0),
            
            # TP/SL Summary
            'risk_management': {
                'stop_loss_percent': results.get('optimal_tp_sl', {}).get('stop_loss', 0.05) * 100,
                'take_profit_percent': results.get('optimal_tp_sl', {}).get('take_profit', 0.10) * 100,
                'risk_reward_ratio': results.get('optimal_tp_sl', {}).get('risk_reward_ratio', 2.0)
            },
            
            # Top indicators
            'top_indicators': results.get('top_indicators', [])[:5],
            
            # Performance comparison
            'vs_default_config': {
                'stop_loss_change': (results.get('optimal_tp_sl', {}).get('stop_loss', 0.05) - 0.05) * 100,
                'take_profit_change': (results.get('optimal_tp_sl', {}).get('take_profit', 0.10) - 0.10) * 100,
                'risk_reward_change': results.get('optimal_tp_sl', {}).get('risk_reward_ratio', 2.0) - 2.0
            }
        }
        
        return summary


def main():
    """
    Demonstration of optimization integration
    """
    print("🔗 OPTIMIZATION INTEGRATION SYSTEM")
    print("=" * 60)
    print("Integrates comprehensive optimization with existing system")
    print("=" * 60)
    print()
    
    # Create integration instance
    integration = OptimizationIntegration()
    
    print("🎯 INTEGRATION FEATURES:")
    print("   ✅ Runs comprehensive optimization")
    print("   ✅ Automatically updates configuration")
    print("   ✅ Saves results for later use")
    print("   ✅ Applies optimized TP/SL parameters")
    print("   ✅ Updates indicator priorities")
    print()
    
    print("💡 USAGE EXAMPLES:")
    print("   1. integration.run_comprehensive_optimization(data)")
    print("   2. integration.apply_optimized_parameters()")
    print("   3. integration.get_optimization_summary()")
    print()
    
    # Check for existing results
    existing_results = integration.load_optimization_results()
    if existing_results:
        print("📊 EXISTING OPTIMIZATION RESULTS FOUND:")
        print("-" * 40)
        summary = integration.get_optimization_summary()
        
        if summary:
            print(f"Date: {summary['optimization_date']}")
            print(f"Score: {summary['performance_score']:.3f}")
            print(f"Stop Loss: {summary['risk_management']['stop_loss_percent']:.1f}%")
            print(f"Take Profit: {summary['risk_management']['take_profit_percent']:.1f}%")
            print(f"Top Indicators: {', '.join(summary['top_indicators'])}")
    else:
        print("📋 No existing optimization results found")
        print("   Run comprehensive_optimization_test.py first")
    
    print("\n🚀 Ready for integration with your main system!")


if __name__ == "__main__":
    main()