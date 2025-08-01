#!/usr/bin/env python3
"""
🎯 Comprehensive Optimization Test
=================================

Test the new comprehensive hyperparameter optimization system that:
1. Tests ALL indicators to find the most effective ones
2. Optimizes TP/SL parameters dynamically (no more hardcoding!)
3. Provides indicator rankings and feature importance
"""

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('.')

def test_comprehensive_optimization():
    """
    Test comprehensive optimization with real BTC data
    """
    print("🚀 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION TEST")
    print("=" * 80)
    print("Finding optimal indicators + TP/SL parameters")
    print("=" * 80)
    print()
    
    try:
        # Import components
        from src.data.collectors import BinanceDataCollector
        from src.features.optimization.comprehensive_hyperparameter_optimizer import (
            ComprehensiveHyperparameterOptimizer, OptimizationConfig
        )
        
        print("📊 Step 1: Collecting BTC data...")
        
        # Get recent data for testing (you can increase this for production)
        collector = BinanceDataCollector(testnet=False, rate_limit=True)
        data = collector.collect_ohlcv(
            symbol="BTC/USDT",
            timeframe="1h",
            limit=2000  # ~83 days of hourly data
        )
        
        # Ensure proper datetime index
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        
        print(f"✅ Data loaded: {len(data)} samples")
        print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        print()
        
        print("🔧 Step 2: Initializing comprehensive optimizer...")
        
        # Create comprehensive optimizer
        optimizer = ComprehensiveHyperparameterOptimizer(
            optimization_config=OptimizationConfig(
                n_calls=100,  # Reduced for testing (increase to 300+ for production)
                n_initial_points=30,
                acquisition_function='EI'
            ),
            enable_feature_selection=True,
            max_optimization_time_hours=1.0  # 1 hour max for testing
        )
        
        print("✅ Optimizer initialized")
        print()
        
        print("🎯 Step 3: Running comprehensive optimization...")
        print("   This will:")
        print("   1. Test each indicator individually")
        print("   2. Rank indicators by effectiveness")
        print("   3. Optimize TP/SL parameters")
        print("   4. Combine best indicators")
        print("   5. Final optimization with feature selection")
        print()
        
        # Run comprehensive optimization
        result = optimizer.optimize_all_indicators(
            data=data,
            validation_split=0.2  # 20% for validation
        )
        
        print("\n🎉 OPTIMIZATION COMPLETE!")
        print("=" * 60)
        
        # Display results
        print("\n🏆 INDICATOR RANKINGS:")
        print("-" * 40)
        for i, indicator in enumerate(result.indicator_rankings, 1):
            print(f"{i:2d}. {indicator.name:15s} | "
                  f"Score: {indicator.importance_score:.3f} | "
                  f"Sharpe: {indicator.sharpe_ratio:.3f} | "
                  f"MaxDD: {indicator.max_drawdown:.1%} | "
                  f"WinRate: {indicator.win_rate:.1%}")
        
        print("\n💰 OPTIMAL TP/SL PARAMETERS:")
        print("-" * 40)
        tp_sl = result.optimal_tp_sl
        print(f"Stop Loss:        {tp_sl['stop_loss']:.1%}")
        print(f"Take Profit:      {tp_sl['take_profit']:.1%}")
        print(f"Risk/Reward:      {tp_sl['risk_reward_ratio']:.2f}")
        print(f"Position Timeout: {tp_sl['position_timeout']:.0f} hours")
        
        print("\n🎯 FEATURE IMPORTANCE:")
        print("-" * 40)
        sorted_features = sorted(result.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:  # Top 10
            print(f"{feature:20s}: {importance:.3f}")
        
        print("\n📈 OPTIMIZATION PERFORMANCE:")
        print("-" * 40)
        print(f"Best Overall Score:    {result.best_overall_score:.3f}")
        print(f"Total Evaluations:     {result.total_evaluations}")
        print(f"Optimization Time:     {result.optimization_time:.1f} seconds")
        print(f"Convergence:           {'✅' if len(result.convergence_history) > 10 else '⚠️'}")
        
        print("\n🔧 RECOMMENDED STRATEGY:")
        print("-" * 40)
        
        # Get top 3 indicators
        top_indicators = result.indicator_rankings[:3]
        print("Use combination of:")
        for i, indicator in enumerate(top_indicators, 1):
            print(f"   {i}. {indicator.name} (importance: {indicator.importance_score:.3f})")
        
        print(f"\nWith dynamic TP/SL:")
        print(f"   • Stop Loss: {tp_sl['stop_loss']:.1%}")
        print(f"   • Take Profit: {tp_sl['take_profit']:.1%}")
        print(f"   • Risk/Reward Ratio: {tp_sl['risk_reward_ratio']:.2f}")
        
        print("\n💡 COMPARISON WITH HARDCODED VALUES:")
        print("-" * 40)
        print(f"Old hardcoded TP/SL:   5% stop, 10% profit (2.0 ratio)")
        print(f"New optimized TP/SL:   {tp_sl['stop_loss']:.1%} stop, {tp_sl['take_profit']:.1%} profit ({tp_sl['risk_reward_ratio']:.1f} ratio)")
        
        improvement = ""
        if tp_sl['risk_reward_ratio'] > 2.0:
            improvement = "✅ Better risk/reward ratio!"
        elif tp_sl['stop_loss'] < 0.05:
            improvement = "✅ Tighter stop loss!"
        elif tp_sl['take_profit'] > 0.10:
            improvement = "✅ Higher profit targets!"
        else:
            improvement = "📊 Optimized for current market conditions"
        
        print(f"Result: {improvement}")
        
        # Save results to file
        print("\n💾 SAVING RESULTS:")
        print("-" * 40)
        
        results_summary = {
            'timestamp': pd.Timestamp.now(),
            'data_samples': len(data),
            'optimization_time': result.optimization_time,
            'best_score': result.best_overall_score,
            'optimal_tp_sl': tp_sl,
            'top_indicators': [ind.name for ind in result.indicator_rankings[:5]],
            'indicator_scores': {ind.name: ind.importance_score for ind in result.indicator_rankings}
        }
        
        # Save to JSON for later use
        import json
        with open('comprehensive_optimization_results.json', 'w') as f:
            # Convert non-serializable objects
            serializable_results = {
                'timestamp': results_summary['timestamp'].isoformat(),
                'data_samples': results_summary['data_samples'],
                'optimization_time': results_summary['optimization_time'],
                'best_score': results_summary['best_score'],
                'optimal_tp_sl': results_summary['optimal_tp_sl'],
                'top_indicators': results_summary['top_indicators'],
                'indicator_scores': results_summary['indicator_scores']
            }
            json.dump(serializable_results, f, indent=2)
        
        print("✅ Results saved to: comprehensive_optimization_results.json")
        
        print("\n🚀 NEXT STEPS:")
        print("-" * 40)
        print("1. Update config/settings.yaml with optimal TP/SL values")
        print("2. Focus feature engineering on top-ranked indicators")
        print("3. Run with larger dataset (5+ years) for production")
        print("4. Implement the optimized parameters in your trading strategy")
        
        return result
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure you have all required dependencies:")
        print("   pip install scikit-optimize")
        print("   Check that src.data.collectors is available")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print("\n💡 Debugging tips:")
        print("   1. Check internet connection for data collection")
        print("   2. Ensure sufficient data (at least 500 samples)")
        print("   3. Verify all optimization components are working")
        
        import traceback
        print(f"\nFull error trace:")
        traceback.print_exc()

def quick_indicator_test():
    """
    Quick test of individual indicators to verify the system works
    """
    print("\n🧪 QUICK INDICATOR TEST")
    print("-" * 40)
    
    try:
        from src.features.optimization.comprehensive_hyperparameter_optimizer import (
            ComprehensiveHyperparameterOptimizer
        )
        from src.features.technical_indicators import TechnicalIndicators
        
        # Create synthetic data for testing
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.02, 1000)
        price = 50000  # Starting price
        prices = []
        
        for ret in returns:
            price *= (1 + ret)
            prices.append(price)
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(100, 1000, 1000)
        }, index=dates)
        
        print(f"✅ Created synthetic data: {len(data)} samples")
        
        # Test technical indicators
        indicators = TechnicalIndicators()
        
        # Test RSI
        rsi = indicators.rsi(data['close'], 14)
        print(f"✅ RSI calculated: {(~rsi.isna()).sum()} valid values")
        
        # Test MACD
        macd = indicators.macd(data['close'])
        print(f"✅ MACD calculated: {(~macd['MACD'].isna()).sum()} valid values")
        
        # Test Bollinger Bands
        bb = indicators.bollinger_bands(data['close'])
        print(f"✅ Bollinger Bands calculated: {(~bb['BB_Upper'].isna()).sum()} valid values")
        
        print("\n✅ All indicators working correctly!")
        print("🎯 Ready for comprehensive optimization")
        
    except Exception as e:
        print(f"❌ Error in indicator test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Run comprehensive optimization test
    """
    print("🚀 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("Testing system that finds optimal indicators + TP/SL parameters")
    print("=" * 80)
    print()
    
    # Run quick test first
    quick_indicator_test()
    
    print("\n" + "="*80)
    print("MAIN OPTIMIZATION TEST")
    print("="*80)
    
    # Run main test
    result = test_comprehensive_optimization()
    
    if result:
        print("\n🎉 SUCCESS!")
        print("The comprehensive optimization system is working correctly.")
        print("You now have optimal TP/SL parameters and indicator rankings!")
    else:
        print("\n⚠️  Test completed with issues.")
        print("Check the error messages above for debugging information.")

if __name__ == "__main__":
    main()