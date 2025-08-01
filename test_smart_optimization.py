"""
🧪 Smart Optimization System Test
=================================

Quick test script to validate the smart optimization system components
work correctly before running the full demonstration.

This script performs rapid validation of:
- Component imports and initialization
- Basic functionality of each module
- Integration between components
- Error handling and fallbacks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all components can be imported successfully."""
    print("🧪 Testing Component Imports...")
    
    try:
        from src.features.optimization import (
            SmartOptimizer,
            ObjectiveFunctions,
            ParameterSpaces,
            WalkForwardAnalyzer,
            MarketRegimeDetector,
            FeatureImportanceFilter
        )
        print("  ✅ All optimization components imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_parameter_spaces():
    """Test parameter space definitions."""
    print("🧪 Testing Parameter Spaces...")
    
    try:
        from src.features.optimization import ParameterSpaces
        
        param_spaces = ParameterSpaces()
        
        # Test RSI space
        rsi_space, rsi_names = param_spaces.get_rsi_space()
        assert len(rsi_space) > 0, "RSI space should not be empty"
        assert len(rsi_names) > 0, "RSI parameter names should not be empty"
        
        # Test MACD space
        macd_space, macd_names = param_spaces.get_macd_space()
        assert len(macd_space) > 0, "MACD space should not be empty"
        
        # Test composite strategy space
        comp_space, comp_names = param_spaces.get_composite_strategy_space()
        assert len(comp_space) > 5, "Composite space should have multiple parameters"
        
        print(f"  ✅ Parameter spaces working - {len(comp_names)} parameters defined")
        return True
    except Exception as e:
        print(f"  ❌ Parameter spaces test failed: {e}")
        return False

def test_objective_functions():
    """Test objective function calculations."""
    print("🧪 Testing Objective Functions...")
    
    try:
        from src.features.optimization import ObjectiveFunctions
        
        # Create synthetic test data
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), periods=100, freq='1H')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 45000 + np.random.normal(0, 100, 100),
            'high': 45000 + np.random.normal(100, 100, 100),
            'low': 45000 + np.random.normal(-100, 100, 100),
            'close': 45000 + np.random.normal(0, 100, 100),
            'volume': np.random.lognormal(10, 1, 100)
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        obj_funcs = ObjectiveFunctions()
        
        # Test parameters
        test_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0
        }
        
        # Test Sharpe ratio calculation
        sharpe = obj_funcs.sharpe_ratio_objective(data, test_params)
        assert isinstance(sharpe, (int, float)), "Sharpe ratio should be numeric"
        
        # Test max drawdown calculation
        drawdown = obj_funcs.max_drawdown_objective(data, test_params)
        assert isinstance(drawdown, (int, float)), "Max drawdown should be numeric"
        
        # Test win rate calculation
        win_rate = obj_funcs.win_rate_objective(data, test_params)
        assert isinstance(win_rate, (int, float)), "Win rate should be numeric"
        assert 0 <= win_rate <= 1, "Win rate should be between 0 and 1"
        
        # Test composite objective
        composite = obj_funcs.composite_objective(data, test_params)
        assert isinstance(composite, (int, float)), "Composite score should be numeric"
        
        print(f"  ✅ Objective functions working - Sharpe: {sharpe:.3f}, Win Rate: {win_rate:.3f}")
        return True
    except Exception as e:
        print(f"  ❌ Objective functions test failed: {e}")
        return False

def test_smart_optimizer():
    """Test basic optimizer functionality."""
    print("🧪 Testing Smart Optimizer...")
    
    try:
        from src.features.optimization import SmartOptimizer, OptimizationConfig, ParameterSpaces, ObjectiveFunctions
        from skopt.space import Integer, Real
        
        # Create minimal test data
        dates = pd.date_range(start=datetime.now() - timedelta(days=2), periods=50, freq='1H')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 45000 + np.random.normal(0, 100, 50),
            'high': 45000 + np.random.normal(100, 100, 50),
            'low': 45000 + np.random.normal(-100, 100, 50),
            'close': 45000 + np.random.normal(0, 100, 50),
            'volume': np.random.lognormal(10, 1, 50)
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        # Configure optimizer for quick test
        config = OptimizationConfig(
            n_calls=5,  # Very few calls for quick test
            n_initial_points=3,
            n_jobs=1
        )
        
        optimizer = SmartOptimizer(config)
        param_spaces = ParameterSpaces()
        obj_funcs = ObjectiveFunctions()
        
        # Simple parameter space for test
        parameter_space = [
            Integer(7, 21, name='rsi_period'),
            Integer(25, 35, name='rsi_oversold')
        ]
        parameter_names = ['rsi_period', 'rsi_oversold']
        
        # Simple objective function
        def test_objective(data, params):
            return obj_funcs.sharpe_ratio_objective(data, params)
        
        # Run quick optimization
        result = optimizer.optimize_indicators(
            data=data,
            parameter_space=parameter_space,
            objective_function=test_objective,
            parameter_names=parameter_names
        )
        
        assert result is not None, "Optimization should return a result"
        assert result.best_params is not None, "Best parameters should be found"
        assert len(result.best_params) == len(parameter_names), "Should have all parameters"
        
        print(f"  ✅ Smart optimizer working - Best score: {result.best_score:.4f}")
        return True
    except Exception as e:
        print(f"  ❌ Smart optimizer test failed: {e}")
        return False

def test_market_regime_detector():
    """Test market regime detection."""
    print("🧪 Testing Market Regime Detector...")
    
    try:
        from src.features.optimization import MarketRegimeDetector
        
        # Create test data with clear regime patterns
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=200, freq='1H')
        np.random.seed(42)
        
        # Create trending data in first half, ranging in second half
        trend_data = np.cumsum(np.random.normal(0.001, 0.01, 100))  # Trending
        range_data = np.random.normal(0, 0.005, 100)  # Ranging
        
        combined_returns = np.concatenate([trend_data, range_data])
        prices = 45000 * np.exp(combined_returns)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 200))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 200))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, 200)
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        detector = MarketRegimeDetector()
        
        # Test regime detection
        regime_result = detector.detect_regimes(data)
        
        assert regime_result is not None, "Regime detection should return a result"
        assert len(regime_result.regimes) > 0, "Should detect some regimes"
        assert regime_result.current_regime is not None, "Should identify current regime"
        
        # Test regime-specific parameters
        regime_params = detector.get_regime_specific_parameters(regime_result.current_regime)
        assert isinstance(regime_params, dict), "Should return parameter dictionary"
        assert len(regime_params) > 0, "Should have some parameters"
        
        print(f"  ✅ Market regime detector working - Current: {regime_result.current_regime.value}")
        return True
    except Exception as e:
        print(f"  ❌ Market regime detector test failed: {e}")
        return False

def test_feature_importance_filter():
    """Test feature importance filtering."""
    print("🧪 Testing Feature Importance Filter...")
    
    try:
        from src.features.optimization import FeatureImportanceFilter, FeatureFilterConfig
        
        # Create test data
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), periods=100, freq='1H')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 45000 + np.random.normal(0, 100, 100),
            'high': 45000 + np.random.normal(100, 100, 100),
            'low': 45000 + np.random.normal(-100, 100, 100),
            'close': 45000 + np.random.normal(0, 100, 100),
            'volume': np.random.lognormal(10, 1, 100)
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        # Configure for quick test
        config = FeatureFilterConfig(
            max_features=10,
            use_advanced_models=False  # Disable to avoid dependency issues
        )
        
        filter_system = FeatureImportanceFilter(config)
        
        # Test indicator rankings
        rankings = filter_system.get_indicator_rankings(data)
        
        assert isinstance(rankings, dict), "Should return rankings dictionary"
        assert len(rankings) > 0, "Should have some indicator rankings"
        
        print(f"  ✅ Feature importance filter working - {len(rankings)} indicators ranked")
        return True
    except Exception as e:
        print(f"  ❌ Feature importance filter test failed: {e}")
        return False

def run_all_tests():
    """Run all component tests."""
    print("🚀 SMART OPTIMIZATION SYSTEM - COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        ("Component Imports", test_imports),
        ("Parameter Spaces", test_parameter_spaces),
        ("Objective Functions", test_objective_functions),
        ("Smart Optimizer", test_smart_optimizer),
        ("Market Regime Detector", test_market_regime_detector),
        ("Feature Importance Filter", test_feature_importance_filter),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 All components working correctly! Ready for full demonstration.")
        print("💡 Run 'python smart_optimization_demo.py' for the complete demo.")
    else:
        print("⚠️  Some components need attention before running the full demo.")
    
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()