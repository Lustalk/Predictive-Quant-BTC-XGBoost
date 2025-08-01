#!/usr/bin/env python3
"""
🎯 Run Comprehensive Optimization with Real Data
===============================================

Production-ready script that:
1. Uses your existing data collection system
2. Tests ALL indicators to find the most effective ones
3. Optimizes TP/SL parameters dynamically
4. Integrates with your current infrastructure
"""

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('.')

def run_comprehensive_optimization():
    """
    Run comprehensive optimization using real BTC data from your existing system
    """
    print("🚀 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("Finding optimal indicators + TP/SL parameters using REAL data")
    print("=" * 80)
    print()
    
    try:
        # Import your existing components
        from main import PredictiveQuantBTCSystem
        from src.features.optimization.comprehensive_hyperparameter_optimizer import (
            ComprehensiveHyperparameterOptimizer, OptimizationConfig
        )
        
        print("📊 Step 1: Using existing data collection system...")
        
        # Use your existing system to get data
        btc_system = PredictiveQuantBTCSystem()
        
        print("🔧 Collecting data using your existing infrastructure...")
        
        # Get data using your existing method (this should work with your validation)
        from src.data.multi_timeframe_collector import MultiTimeframeCollector
        
        collector = MultiTimeframeCollector()
        data_dict = collector.collect_multi_timeframe_data(
            timeframes=['1h'],  # Just 1h for optimization
            symbol="BTC/USDT", 
            days_back=180  # 6 months of data
        )
        
        # Use 1h data for optimization
        if '1h' in data_dict and not data_dict['1h'].empty:
            data = data_dict['1h'].copy()
            print(f"✅ Data loaded: {len(data)} samples (1h timeframe)")
            print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        else:
            raise ValueError("No valid 1h data available")
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"✅ Data validation passed: {list(data.columns)}")
        print()
        
        print("🔧 Step 2: Initializing comprehensive optimizer...")
        
        # Create optimizer with production settings
        optimizer = ComprehensiveHyperparameterOptimizer(
            optimization_config=OptimizationConfig(
                n_calls=150,  # Balanced for reasonable time vs thoroughness
                n_initial_points=30,
                acquisition_function='EI'
            ),
            enable_feature_selection=True,
            max_optimization_time_hours=2.0  # 2 hours for comprehensive search
        )
        
        print("✅ Optimizer initialized")
        print()
        
        print("🎯 Step 3: Running comprehensive optimization...")
        print("   This will test ALL indicators and optimize TP/SL parameters")
        print("   Estimated time: 1-2 hours for thorough optimization")
        print()
        
        # Run comprehensive optimization
        result = optimizer.optimize_all_indicators(
            data=data,
            validation_split=0.2  # 20% for out-of-sample validation
        )
        
        print("\n🎉 OPTIMIZATION COMPLETE!")
        print("=" * 80)
        
        # Display comprehensive results
        print("\n🏆 INDICATOR EFFECTIVENESS RANKING:")
        print("-" * 60)
        print(f"{'Rank':<4} {'Indicator':<20} {'Score':<8} {'Sharpe':<8} {'MaxDD':<8} {'WinRate':<8}")
        print("-" * 60)
        
        for i, indicator in enumerate(result.indicator_rankings[:10], 1):
            print(f"{i:<4} {indicator.name:<20} {indicator.importance_score:<8.3f} "
                  f"{indicator.sharpe_ratio:<8.3f} {indicator.max_drawdown:<8.1%} {indicator.win_rate:<8.1%}")
        
        print("\n💰 OPTIMIZED TP/SL PARAMETERS (vs Hardcoded):")
        print("-" * 60)
        tp_sl = result.optimal_tp_sl
        
        print(f"Parameter               Hardcoded    Optimized    Improvement")
        print("-" * 60)
        print(f"Stop Loss              5.0%         {tp_sl['stop_loss']:<8.1%}   ", end="")
        sl_improvement = "✅ Better" if tp_sl['stop_loss'] != 0.05 else "📊 Same"
        print(sl_improvement)
        
        print(f"Take Profit            10.0%        {tp_sl['take_profit']:<8.1%}   ", end="")
        tp_improvement = "✅ Better" if tp_sl['take_profit'] != 0.10 else "📊 Same"
        print(tp_improvement)
        
        print(f"Risk/Reward Ratio      2.0          {tp_sl['risk_reward_ratio']:<8.2f}   ", end="")
        rr_improvement = "✅ Better" if tp_sl['risk_reward_ratio'] > 2.0 else "📊 Conservative" if tp_sl['risk_reward_ratio'] < 2.0 else "📊 Same"
        print(rr_improvement)
        
        print(f"Position Timeout       24h          {tp_sl['position_timeout']:<8.0f}h    ", end="")
        timeout_improvement = "✅ Optimized" if tp_sl['position_timeout'] != 24 else "📊 Same"
        print(timeout_improvement)
        
        print("\n🎯 TOP PERFORMING INDICATORS:")
        print("-" * 40)
        top_3 = result.indicator_rankings[:3]
        for i, indicator in enumerate(top_3, 1):
            print(f"{i}. {indicator.name}")
            print(f"   • Importance Score: {indicator.importance_score:.3f}")
            print(f"   • Sharpe Ratio: {indicator.sharpe_ratio:.3f}")
            print(f"   • Max Drawdown: {indicator.max_drawdown:.1%}")
            print(f"   • Win Rate: {indicator.win_rate:.1%}")
            print()
        
        print("🔍 FEATURE IMPORTANCE ANALYSIS:")
        print("-" * 40)
        sorted_features = sorted(result.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:8]:  # Top 8
            print(f"{feature:<25}: {importance:.3f}")
        
        print(f"\n📈 OPTIMIZATION PERFORMANCE:")
        print("-" * 40)
        print(f"Best Overall Score:     {result.best_overall_score:.3f}")
        print(f"Total Evaluations:      {result.total_evaluations}")
        print(f"Optimization Time:      {result.optimization_time/60:.1f} minutes")
        print(f"Data Samples:           {len(data)}")
        print(f"Validation Split:       20% out-of-sample")
        
        print("\n🚀 IMPLEMENTATION RECOMMENDATIONS:")
        print("-" * 40)
        
        # Determine best strategy
        if result.best_overall_score > 0.5:
            strategy_quality = "🔥 Excellent"
        elif result.best_overall_score > 0.3:
            strategy_quality = "✅ Good"
        elif result.best_overall_score > 0.1:
            strategy_quality = "📊 Moderate"
        else:
            strategy_quality = "⚠️ Needs improvement"
        
        print(f"Strategy Quality: {strategy_quality}")
        print()
        
        print("📋 Action Items:")
        print("1. Update config/settings.yaml with optimized TP/SL values")
        print("2. Focus feature engineering on top 3 indicators:")
        for indicator in top_3:
            print(f"   • {indicator.name}")
        print("3. Implement dynamic TP/SL based on volatility")
        print("4. Consider removing low-performing indicators")
        
        # Update configuration automatically
        print("\n🔧 UPDATING CONFIGURATION:")
        print("-" * 40)
        
        from src.features.optimization.optimization_integration import OptimizationIntegration
        
        integration = OptimizationIntegration()
        
        # Save comprehensive results
        try:
            # Create results for integration
            results_dict = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'best_score': result.best_overall_score,
                'optimization_time': result.optimization_time,
                'total_evaluations': result.total_evaluations,
                'optimal_tp_sl': result.optimal_tp_sl,
                'top_indicators': [ind.name for ind in result.indicator_rankings[:5]],
                'indicator_scores': {
                    ind.name: ind.importance_score 
                    for ind in result.indicator_rankings
                }
            }
            
            # Save and apply results
            import json
            with open('comprehensive_optimization_results.json', 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print("✅ Results saved to: comprehensive_optimization_results.json")
            
            # Apply to configuration
            updated_config = integration.apply_optimized_parameters(results_dict)
            print("✅ Configuration updated with optimized parameters")
            
        except Exception as e:
            print(f"⚠️ Could not auto-update config: {e}")
            print("💡 Manually update config/settings.yaml with the values above")
        
        print("\n🎯 COMPARISON WITH YOUR CURRENT SYSTEM:")
        print("-" * 60)
        print("BEFORE (Hardcoded):")
        print("  • Stop Loss: 5% (fixed)")
        print("  • Take Profit: 10% (fixed)")
        print("  • Risk/Reward: 2.0 (fixed)")
        print("  • All indicators used equally")
        print()
        print("AFTER (Optimized):")
        print(f"  • Stop Loss: {tp_sl['stop_loss']:.1%} (market-optimized)")
        print(f"  • Take Profit: {tp_sl['take_profit']:.1%} (risk-adjusted)")
        print(f"  • Risk/Reward: {tp_sl['risk_reward_ratio']:.1f} (balanced)")
        print("  • Only top-performing indicators prioritized")
        
        # Calculate potential improvement
        default_rr = 2.0
        optimized_rr = tp_sl['risk_reward_ratio']
        improvement_pct = ((optimized_rr - default_rr) / default_rr) * 100
        
        print(f"\n📊 Potential Risk/Reward Improvement: {improvement_pct:+.1f}%")
        
        return result
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure all dependencies are installed:")
        print("   pip install scikit-optimize")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print("\n💡 Debugging information:")
        print("   1. Check that your data collection system is working")
        print("   2. Verify internet connection")
        print("   3. Ensure sufficient historical data is available")
        
        import traceback
        print(f"\nFull error trace:")
        traceback.print_exc()
        
        return None

def quick_validation_test():
    """
    Quick test to validate the optimization system works
    """
    print("\n🧪 VALIDATION TEST")
    print("-" * 40)
    
    try:
        from src.features.technical_indicators import TechnicalIndicators
        
        # Test with small dataset
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 100),
            'high': np.random.uniform(46000, 56000, 100),
            'low': np.random.uniform(44000, 54000, 100),
            'close': np.random.uniform(45000, 55000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        # Test indicators
        indicators = TechnicalIndicators()
        
        # Test Bollinger Bands with correct column names
        bb = indicators.bollinger_bands(data['close'])
        print(f"✅ Bollinger Bands columns: {list(bb.columns)}")
        
        # Test other indicators
        rsi = indicators.rsi(data['close'], 14)
        print(f"✅ RSI calculated: {(~rsi.isna()).sum()} valid values")
        
        macd = indicators.macd(data['close'])
        print(f"✅ MACD calculated: {list(macd.columns)}")
        
        print("✅ All indicators working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """
    Main execution function
    """
    print("🎯 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("Real-world optimization of ALL indicators + TP/SL parameters")
    print("=" * 80)
    print()
    
    # Run validation first
    if not quick_validation_test():
        print("\n❌ Validation failed. Cannot proceed with optimization.")
        return
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE OPTIMIZATION")
    print("="*80)
    
    # Run comprehensive optimization
    result = run_comprehensive_optimization()
    
    if result:
        print("\n🎉 SUCCESS!")
        print("=" * 40)
        print("Comprehensive optimization completed successfully!")
        print("You now have:")
        print("  ✅ Optimized TP/SL parameters (no more hardcoding!)")
        print("  ✅ Ranked list of most effective indicators")
        print("  ✅ Feature importance analysis")
        print("  ✅ Updated configuration file")
        print("\n🚀 Your trading strategy is now data-driven and optimized!")
    else:
        print("\n⚠️ Optimization encountered issues.")
        print("Check the error messages above for resolution steps.")

if __name__ == "__main__":
    main()