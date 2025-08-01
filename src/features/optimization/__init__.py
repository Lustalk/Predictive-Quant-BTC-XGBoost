"""
Smart Optimization System for Technical Indicators
================================================

Modern quantitative finance optimization suite using:
- Bayesian Optimization with scikit-optimize
- Multi-objective optimization (Sharpe + Drawdown + Win Rate)
- Walk-forward analysis for robustness testing
- Market regime detection for adaptive parameters
- Ensemble-based feature importance filtering
"""

from .smart_optimizer import SmartOptimizer, OptimizationConfig
from .objective_functions import ObjectiveFunctions
from .parameter_spaces import ParameterSpaces
from .walk_forward_analyzer import WalkForwardAnalyzer
from .market_regime_detector import MarketRegimeDetector
from .feature_importance_filter import FeatureImportanceFilter, FeatureFilterConfig

__all__ = [
    "SmartOptimizer",
    "OptimizationConfig",
    "ObjectiveFunctions",
    "ParameterSpaces",
    "WalkForwardAnalyzer",
    "MarketRegimeDetector",
    "FeatureImportanceFilter",
    "FeatureFilterConfig",
]
