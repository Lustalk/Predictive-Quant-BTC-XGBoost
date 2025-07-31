#!/usr/bin/env python3
"""
Demonstration script for the XGBoost cryptocurrency trading system.
Shows end-to-end functionality with real Binance data.

This script demonstrates:
1. Real data collection from Binance
2. Comprehensive feature engineering (50+ features)
3. XGBoost model training and optimization
4. Basic backtesting with performance metrics
5. Feature importance analysis

Run this script to see the complete system in action.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
from src.data.collectors import BinanceDataCollector
from src.data.storage import TimeSeriesStorage
from src.features.feature_engine import FeatureEngine
from src.features.technical_indicators import TechnicalIndicators
from src.utils.logging import setup_logging

# ML imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def create_target_variable(data: pd.DataFrame, horizon: int = 4) -> pd.Series:
    """
    Create binary classification target: will price go up in next N hours?
    
    Args:
        data: DataFrame with OHLCV data
        horizon: Number of periods to look ahead
        
    Returns:
        Binary target series (1 = price up, 0 = price down/same)
    """
    future_price = data['close'].shift(-horizon)
    current_price = data['close']
    
    # 1 if price goes up, 0 if price goes down or stays same
    target = (future_price > current_price).astype(int)
    
    # Remove the last N rows since we can't predict their future
    target = target[:-horizon]
    
    return target


def run_xgboost_demo():
    """Run the complete XGBoost cryptocurrency trading system demo."""
    
    print("🚀 Starting XGBoost Cryptocurrency Trading System Demo")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Step 1: Data Collection
        print("\n📊 Step 1: Collecting Real BTC Data from Binance...")
        collector = BinanceDataCollector(testnet=False, rate_limit=True)
        
        # Collect last 30 days of hourly data (no API keys needed for public data)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"   Fetching BTC/USDT data from {start_date} to {end_date}")
        data = collector.collect_ohlcv(
            symbol="BTC/USDT",
            timeframe="1h", 
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"   ✅ Collected {len(data)} hourly candles")
        print(f"   📈 Price range: ${data['close'].min():.0f} - ${data['close'].max():.0f}")
        
        # Step 2: Feature Engineering
        print("\n🔧 Step 2: Engineering Features...")
        feature_engine = FeatureEngine()
        
        # Set timestamp as index for time-based features
        data_indexed = data.set_index('timestamp')
        
        # Create all features
        features = feature_engine.create_all_features(data_indexed)
        print(f"   ✅ Created {len(features.columns)} features")
        
        # Memory usage report
        memory_report = feature_engine.memory_usage_report(features)
        print(f"   💾 Memory usage: {memory_report['total_memory_mb']:.1f} MB")
        
        # Step 3: Target Variable Creation
        print("\n🎯 Step 3: Creating Target Variable...")
        target = create_target_variable(data_indexed, horizon=4)
        
        # Align features and target
        min_length = min(len(features), len(target))
        features_aligned = features.iloc[:min_length]
        target_aligned = target.iloc[:min_length]
        
        print(f"   ✅ Target created: predicting 4-hour price direction")
        print(f"   📊 Class distribution: {target_aligned.value_counts().to_dict()}")
        print(f"   ⚖️  Class balance: {target_aligned.mean():.1%} positive")
        
        # Step 4: Data Splitting
        print("\n✂️  Step 4: Splitting Data...")
        
        # Use time-based split to prevent data leakage
        split_point = int(len(features_aligned) * 0.8)
        
        X_train = features_aligned.iloc[:split_point]
        X_test = features_aligned.iloc[split_point:]
        y_train = target_aligned.iloc[:split_point]
        y_test = target_aligned.iloc[split_point:]
        
        print(f"   📈 Training set: {len(X_train)} samples")
        print(f"   📉 Test set: {len(X_test)} samples")
        
        # Step 5: Feature Selection
        print("\n🎯 Step 5: Selecting Best Features...")
        top_features = feature_engine.select_features(X_train, y_train, method='mutual_info', top_k=30)
        
        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]
        
        print(f"   ✅ Selected top {len(top_features)} features")
        print(f"   🔝 Top 5 features: {top_features[:5]}")
        
        # Step 6: Model Training
        print("\n🤖 Step 6: Training XGBoost Model...")
        
        # XGBoost parameters optimized for binary classification
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',  # Faster training
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**xgb_params)
        
        # Train model
        model.fit(
            X_train_selected, y_train,
            eval_set=[(X_test_selected, y_test)],
            verbose=False
        )
        
        print(f"   ✅ Model trained with {model.n_estimators} trees")
        
        # Step 7: Model Evaluation
        print("\n📊 Step 7: Evaluating Model Performance...")
        
        # Predictions
        y_pred_train = model.predict(X_train_selected)
        y_pred_test = model.predict(X_test_selected)
        y_pred_proba_test = model.predict_proba(X_test_selected)[:, 1]
        
        # Performance metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"   📈 Training Accuracy: {train_accuracy:.1%}")
        print(f"   📉 Test Accuracy: {test_accuracy:.1%}")
        
        # Detailed classification report
        print("\n   📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=['Down', 'Up']))
        
        # Step 8: Feature Importance Analysis
        print("\n🔍 Step 8: Analyzing Feature Importance...")
        feature_importance = feature_engine.get_feature_importance(model, top_features)
        
        print("\n   🏆 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"   {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        # Step 9: Basic Backtesting
        print("\n📈 Step 9: Running Basic Backtest...")
        
        # Simple strategy: Buy when model predicts up (>60% confidence)
        confidence_threshold = 0.6
        signals = (y_pred_proba_test > confidence_threshold).astype(int)
        
        # Calculate returns
        test_data = data_indexed.iloc[split_point:split_point+len(y_test)]
        returns = test_data['close'].pct_change().fillna(0)
        
        # Strategy returns (only trade when signal is strong)
        signals_series = pd.Series(signals, index=returns.index)
        strategy_returns = returns * signals_series.shift(1).fillna(0)  # Lag signals by 1
        buy_and_hold_returns = returns
        
        # Performance metrics
        total_strategy_return = (1 + strategy_returns).prod() - 1
        total_bnh_return = (1 + buy_and_hold_returns).prod() - 1
        
        strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(24*365) if strategy_returns.std() > 0 else 0
        bnh_sharpe = buy_and_hold_returns.mean() / buy_and_hold_returns.std() * np.sqrt(24*365) if buy_and_hold_returns.std() > 0 else 0
        
        num_trades = signals.sum()
        win_rate = (strategy_returns > 0).sum() / num_trades if num_trades > 0 else 0
        
        print(f"   📊 Backtest Results (Test Period):")
        print(f"   💰 Strategy Return: {total_strategy_return:.1%}")
        print(f"   📈 Buy & Hold Return: {total_bnh_return:.1%}")
        print(f"   ⚡ Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"   📈 Buy & Hold Sharpe: {bnh_sharpe:.2f}")
        print(f"   🎯 Number of Trades: {num_trades}")
        print(f"   🏆 Win Rate: {win_rate:.1%}")
        
        # Step 10: Visualization
        print("\n📊 Step 10: Creating Visualizations...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('XGBoost Cryptocurrency Trading System Results', fontsize=16)
            
            # Plot 1: Feature Importance
            top_10_features = feature_importance.head(10)
            axes[0, 0].barh(range(len(top_10_features)), top_10_features['importance'])
            axes[0, 0].set_yticks(range(len(top_10_features)))
            axes[0, 0].set_yticklabels(top_10_features['feature'])
            axes[0, 0].set_title('Top 10 Feature Importance')
            axes[0, 0].set_xlabel('Importance')
            
            # Plot 2: Prediction Confidence Distribution
            axes[0, 1].hist(y_pred_proba_test, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(confidence_threshold, color='red', linestyle='--', label=f'Threshold ({confidence_threshold})')
            axes[0, 1].set_title('Prediction Confidence Distribution')
            axes[0, 1].set_xlabel('Predicted Probability')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            
            # Plot 3: Cumulative Returns
            strategy_cumret = (1 + strategy_returns).cumprod()
            bnh_cumret = (1 + buy_and_hold_returns).cumprod()
            
            axes[1, 0].plot(strategy_cumret.index, strategy_cumret.values, label='Strategy', linewidth=2)
            axes[1, 0].plot(bnh_cumret.index, bnh_cumret.values, label='Buy & Hold', linewidth=2)
            axes[1, 0].set_title('Cumulative Returns Comparison')
            axes[1, 0].set_ylabel('Cumulative Return')
            axes[1, 0].legend()
            
            # Plot 4: Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_test)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[1, 1])
            axes[1, 1].set_title('Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig('xgboost_results.png', dpi=300, bbox_inches='tight')
            print("   ✅ Saved visualization as 'xgboost_results.png'")
            
        except Exception as e:
            print(f"   ⚠️  Could not create visualizations: {e}")
        
        # Final Summary
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📊 SYSTEM PERFORMANCE SUMMARY:")
        print(f"   🔢 Features Created: {len(features.columns)}")
        print(f"   🎯 Selected Features: {len(top_features)}")
        print(f"   🤖 Model Accuracy: {test_accuracy:.1%}")
        print(f"   💰 Strategy Return: {total_strategy_return:.1%}")
        print(f"   📈 vs Buy & Hold: {total_strategy_return - total_bnh_return:+.1%}")
        print(f"   ⚡ Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"   🎯 Win Rate: {win_rate:.1%}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Optimize hyperparameters using grid search")
        print(f"   2. Implement walk-forward validation")
        print(f"   3. Add risk management (stop loss, position sizing)")
        print(f"   4. Test on different timeframes")
        print(f"   5. Add ensemble methods (LightGBM + XGBoost)")
        
        return {
            'model': model,
            'features': top_features,
            'test_accuracy': test_accuracy,
            'strategy_return': total_strategy_return,
            'sharpe_ratio': strategy_sharpe,
            'win_rate': win_rate,
            'num_trades': num_trades
        }
        
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_xgboost_demo()
    
    if results:
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Key metrics: Accuracy={results['test_accuracy']:.1%}, "
              f"Return={results['strategy_return']:.1%}, "
              f"Sharpe={results['sharpe_ratio']:.2f}")
    else:
        print(f"\n❌ Demo failed. Check the logs above.")