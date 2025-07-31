#!/usr/bin/env python3
"""
Large Dataset XGBoost Cryptocurrency Trading System Demo.
Demonstrates the system working with 3-5 years of high-quality data across multiple timeframes.

This script shows:
1. Multi-source data collection (3-5 years of data)
2. Gap detection and filling from multiple exchanges
3. Comprehensive data quality validation
4. XGBoost training on large datasets (5000+ samples)
5. Multi-timeframe analysis (1h, 4h, 30m)
6. Advanced performance metrics and visualization
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
from src.data.multi_source_collector import EnhancedDataCollector
from src.features.feature_engine import FeatureEngine
from src.utils.logging import setup_logging

# ML imports
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def create_target_variable(data: pd.DataFrame, horizon: int = 4) -> pd.Series:
    """Create binary classification target for price direction prediction."""
    future_price = data['close'].shift(-horizon)
    current_price = data['close']
    target = (future_price > current_price).astype(int)
    return target[:-horizon]


def train_xgboost_model(X_train: pd.DataFrame, 
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> Dict[str, Any]:
    """Train XGBoost model with optimized parameters for large datasets."""
    
    # Optimized parameters for large datasets
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'verbosity': 0,
        'n_jobs': -1  # Use all CPU cores
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    auc_score = roc_auc_score(y_test, y_pred_proba_test)
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'auc_score': auc_score,
        'predictions': y_pred_test,
        'probabilities': y_pred_proba_test
    }


def run_comprehensive_backtest(data: pd.DataFrame, 
                              predictions: np.ndarray, 
                              probabilities: np.ndarray,
                              timeframe: str) -> Dict[str, float]:
    """Run comprehensive backtesting with multiple strategies."""
    
    # Calculate returns
    returns = data['close'].pct_change().fillna(0)
    
    # Strategy 1: High confidence trades (>70%)
    high_conf_signals = pd.Series((probabilities > 0.7).astype(int), index=returns.index)
    high_conf_returns = returns * high_conf_signals.shift(1).fillna(0)
    
    # Strategy 2: Medium confidence trades (>60%)
    med_conf_signals = pd.Series((probabilities > 0.6).astype(int), index=returns.index)
    med_conf_returns = returns * med_conf_signals.shift(1).fillna(0)
    
    # Buy and hold
    buy_hold_returns = returns
    
    # Performance metrics
    def calculate_metrics(strategy_returns, name):
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 24) if strategy_returns.std() > 0 else 0
        max_drawdown = calculate_max_drawdown(strategy_returns)
        win_rate = (strategy_returns > 0).mean()
        
        return {
            f'{name}_total_return': total_return,
            f'{name}_sharpe_ratio': sharpe_ratio,
            f'{name}_max_drawdown': max_drawdown,
            f'{name}_win_rate': win_rate
        }
    
    results = {}
    results.update(calculate_metrics(high_conf_returns, 'high_conf'))
    results.update(calculate_metrics(med_conf_returns, 'med_conf'))
    results.update(calculate_metrics(buy_hold_returns, 'buy_hold'))
    
    # Trade counts
    results['high_conf_trades'] = high_conf_signals.sum()
    results['med_conf_trades'] = med_conf_signals.sum()
    results['timeframe'] = timeframe
    
    return results


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def run_large_dataset_demo():
    """Run the comprehensive large dataset demo."""
    
    print("🚀 Starting Large Dataset XGBoost Cryptocurrency Trading System Demo")
    print("=" * 80)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Step 1: Data Collection
        print("\n📊 Step 1: Collecting Multi-Year, Multi-Source Data...")
        print("This may take several minutes as we collect 3-5 years of data...")
        
        collector = EnhancedDataCollector()
        
        # Collect comprehensive dataset
        collection_result = collector.collect_production_dataset(
            symbol="BTC/USDT",
            years_back=4,  # 4 years of data
            timeframes=["1h", "4h"],  # Multiple timeframes
            min_samples_per_timeframe=5000
        )
        
        datasets = collection_result['datasets']
        quality_report = collection_result['quality_report']
        
        print(f"\n✅ Data Collection Complete!")
        print(f"📊 Quality Score: {quality_report.get('average_quality_score', 0):.1f}/100")
        print(f"🏆 Overall Quality: {quality_report.get('overall_quality', 'unknown').title()}")
        
        # Display dataset information
        for timeframe, data in datasets.items():
            tf_report = quality_report['timeframes'][timeframe]
            print(f"\n📈 {timeframe.upper()} Dataset:")
            print(f"   Samples: {len(data):,}")
            print(f"   Date range: {tf_report['date_range']['start'].date()} to {tf_report['date_range']['end'].date()}")
            print(f"   Days covered: {tf_report['date_range']['days']:,}")
            print(f"   Completeness: {tf_report['completeness_pct']:.1f}%")
            print(f"   Quality score: {tf_report['quality_score']:.1f}/100")
        
        # Step 2: Feature Engineering for Large Datasets
        print("\n🔧 Step 2: Advanced Feature Engineering...")
        feature_engine = FeatureEngine()
        
        processed_datasets = {}
        
        for timeframe, data in datasets.items():
            print(f"\n   Processing {timeframe} features...")
            
            # Set timestamp as index
            data_indexed = data.set_index('timestamp')
            
            # Create comprehensive features
            features = feature_engine.create_all_features(data_indexed)
            
            # Memory optimization for large datasets
            memory_report = feature_engine.memory_usage_report(features)
            print(f"   💾 Memory usage: {memory_report['total_memory_mb']:.1f} MB")
            
            # Create target variable
            target = create_target_variable(data_indexed, horizon=4)
            
            # Align features and target
            min_length = min(len(features), len(target))
            features_aligned = features.iloc[:min_length]
            target_aligned = target.iloc[:min_length]
            
            processed_datasets[timeframe] = {
                'features': features_aligned,
                'target': target_aligned,
                'raw_data': data_indexed.iloc[:min_length]
            }
            
            print(f"   ✅ {timeframe}: {len(features_aligned)} features, {len(features_aligned):,} samples")
            print(f"   📊 Target balance: {target_aligned.mean():.1%} positive")
        
        # Step 3: Multi-Timeframe Model Training
        print("\n🤖 Step 3: Training XGBoost Models on Large Datasets...")
        
        model_results = {}
        
        for timeframe, dataset in processed_datasets.items():
            print(f"\n   Training {timeframe} model...")
            
            features = dataset['features']
            target = dataset['target']
            
            # Feature selection for large datasets
            top_features = feature_engine.select_features(
                features, target, method='mutual_info', top_k=50
            )
            
            X = features[top_features]
            y = target
            
            # Time-based split (80/20)
            split_point = int(len(X) * 0.8)
            X_train = X.iloc[:split_point]
            X_test = X.iloc[split_point:]
            y_train = y.iloc[:split_point]
            y_test = y.iloc[split_point:]
            
            print(f"   📈 Training: {len(X_train):,} samples")
            print(f"   📉 Testing: {len(X_test):,} samples")
            
            # Train model
            model_result = train_xgboost_model(X_train, y_train, X_test, y_test)
            
            print(f"   ✅ Training accuracy: {model_result['train_accuracy']:.1%}")
            print(f"   📊 Test accuracy: {model_result['test_accuracy']:.1%}")
            print(f"   🎯 AUC Score: {model_result['auc_score']:.3f}")
            
            model_results[timeframe] = {
                **model_result,
                'features': top_features,
                'test_data': dataset['raw_data'].iloc[split_point:split_point+len(X_test)]
            }
        
        # Step 4: Comprehensive Backtesting
        print("\n📈 Step 4: Multi-Timeframe Backtesting...")
        
        backtest_results = {}
        
        for timeframe, result in model_results.items():
            print(f"\n   Backtesting {timeframe} strategy...")
            
            test_data = result['test_data']
            predictions = result['predictions']
            probabilities = result['probabilities']
            
            backtest = run_comprehensive_backtest(
                test_data, predictions, probabilities, timeframe
            )
            
            backtest_results[timeframe] = backtest
            
            print(f"   💰 High Confidence Strategy:")
            print(f"      Return: {backtest['high_conf_total_return']:.1%}")
            print(f"      Sharpe: {backtest['high_conf_sharpe_ratio']:.2f}")
            print(f"      Max DD: {backtest['high_conf_max_drawdown']:.1%}")
            print(f"      Trades: {backtest['high_conf_trades']:,}")
            
            print(f"   📈 vs Buy & Hold: {backtest['high_conf_total_return'] - backtest['buy_hold_total_return']:+.1%}")
        
        # Step 5: Performance Visualization
        print("\n📊 Step 5: Creating Comprehensive Visualizations...")
        
        try:
            fig, axes = plt.subplots(3, 2, figsize=(20, 18))
            fig.suptitle('Large Dataset XGBoost Trading System Results', fontsize=16)
            
            timeframes = list(model_results.keys())
            
            # Plot 1: Model Performance Comparison
            metrics = ['test_accuracy', 'auc_score']
            tf_names = []
            accuracy_scores = []
            auc_scores = []
            
            for tf in timeframes:
                tf_names.append(tf.upper())
                accuracy_scores.append(model_results[tf]['test_accuracy'])
                auc_scores.append(model_results[tf]['auc_score'])
            
            x = np.arange(len(tf_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, accuracy_scores, width, label='Accuracy', alpha=0.7)
            axes[0, 0].bar(x + width/2, auc_scores, width, label='AUC Score', alpha=0.7)
            axes[0, 0].set_xlabel('Timeframe')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Model Performance by Timeframe')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(tf_names)
            axes[0, 0].legend()
            
            # Plot 2: Strategy Returns Comparison
            strategy_returns = []
            bnh_returns = []
            
            for tf in timeframes:
                strategy_returns.append(backtest_results[tf]['high_conf_total_return'])
                bnh_returns.append(backtest_results[tf]['buy_hold_total_return'])
            
            axes[0, 1].bar(tf_names, strategy_returns, alpha=0.7, label='Strategy')
            axes[0, 1].bar(tf_names, bnh_returns, alpha=0.7, label='Buy & Hold')
            axes[0, 1].set_ylabel('Total Return')
            axes[0, 1].set_title('Strategy vs Buy & Hold Returns')
            axes[0, 1].legend()
            
            # Plot 3 & 4: Feature Importance for each timeframe
            for i, tf in enumerate(timeframes):
                if i < 2:  # Only plot first 2 timeframes
                    model = model_results[tf]['model']
                    features = model_results[tf]['features']
                    
                    importance = feature_engine.get_feature_importance(model, features)
                    top_10 = importance.head(10)
                    
                    axes[1, i].barh(range(len(top_10)), top_10['importance'])
                    axes[1, i].set_yticks(range(len(top_10)))
                    axes[1, i].set_yticklabels(top_10['feature'])
                    axes[1, i].set_title(f'{tf.upper()} Top Features')
                    axes[1, i].set_xlabel('Importance')
            
            # Plot 5: Dataset Quality Metrics
            quality_scores = []
            completeness = []
            
            for tf in timeframes:
                tf_report = quality_report['timeframes'][tf]
                quality_scores.append(tf_report['quality_score'])
                completeness.append(tf_report['completeness_pct'])
            
            axes[2, 0].bar(tf_names, quality_scores, alpha=0.7, color='green')
            axes[2, 0].set_ylabel('Quality Score')
            axes[2, 0].set_title('Data Quality by Timeframe')
            axes[2, 0].set_ylim(0, 100)
            
            # Plot 6: Sample Counts
            sample_counts = [len(datasets[tf]) for tf in timeframes]
            
            axes[2, 1].bar(tf_names, sample_counts, alpha=0.7, color='blue')
            axes[2, 1].set_ylabel('Sample Count')
            axes[2, 1].set_title('Dataset Size by Timeframe')
            
            plt.tight_layout()
            plt.savefig('large_dataset_results.png', dpi=300, bbox_inches='tight')
            print("   ✅ Saved comprehensive visualization as 'large_dataset_results.png'")
            
        except Exception as e:
            print(f"   ⚠️  Could not create visualizations: {e}")
        
        # Final Summary
        print("\n" + "="*80)
        print("🎉 LARGE DATASET DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n📊 DATASET SUMMARY:")
        total_samples = sum(len(data) for data in datasets.values())
        print(f"   📈 Total Samples Collected: {total_samples:,}")
        print(f"   📅 Data Span: {quality_report['timeframes'][timeframes[0]]['date_range']['days']:,} days")
        print(f"   🏆 Average Quality Score: {quality_report.get('average_quality_score', 0):.1f}/100")
        
        print(f"\n🤖 MODEL PERFORMANCE:")
        for tf in timeframes:
            result = model_results[tf]
            backtest = backtest_results[tf]
            print(f"   {tf.upper()} Model:")
            print(f"      Accuracy: {result['test_accuracy']:.1%}")
            print(f"      AUC Score: {result['auc_score']:.3f}")
            print(f"      Strategy Return: {backtest['high_conf_total_return']:.1%}")
            print(f"      Sharpe Ratio: {backtest['high_conf_sharpe_ratio']:.2f}")
        
        print(f"\n🎯 NEXT STEPS FOR PRODUCTION:")
        print(f"   1. Implement real-time data updates")
        print(f"   2. Add ensemble methods across timeframes")
        print(f"   3. Implement dynamic position sizing")
        print(f"   4. Add regime detection")
        print(f"   5. Deploy live trading system")
        
        return {
            'datasets': datasets,
            'quality_report': quality_report,
            'model_results': model_results,
            'backtest_results': backtest_results
        }
        
    except Exception as e:
        print(f"❌ Error in large dataset demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_large_dataset_demo()
    
    if results:
        total_samples = sum(len(data) for data in results['datasets'].values())
        avg_quality = results['quality_report'].get('average_quality_score', 0)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Collected {total_samples:,} samples with {avg_quality:.1f}/100 quality score")
    else:
        print(f"\n❌ Demo failed. Check the logs above.")