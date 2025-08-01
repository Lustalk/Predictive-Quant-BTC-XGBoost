#!/usr/bin/env python3
"""
🚀 PREDICTIVE QUANT BTC-XGBOOST - UNIFIED MAIN SYSTEM
====================================================

A comprehensive cryptocurrency trading prediction system using XGBoost
with advanced feature engineering, technical indicators, and diagnostic analysis.

Usage:
    python main.py

Features:
    ✅ Real-time BTC data collection from Binance
    ✅ 222 advanced features with 20+ technical indicators  
    ✅ XGBoost model with intelligent feature selection
    ✅ Comprehensive backtesting and performance analysis
    ✅ Advanced diagnostic visualizations revealing model thinking
    ✅ Automated improvement recommendations
    ✅ Professional trading dashboard outputs

Author: Data Science Expert
License: MIT
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Core system imports
from src.data.collectors import BinanceDataCollector
from src.features.feature_engine import FeatureEngine
from src.utils.config import get_config
from src.utils.logging import setup_logging
from src.utils.visualizations import TradingVisualizer
from src.utils.diagnostic_visualizations import XGBoostDiagnosticVisualizer

# Machine learning imports
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class PredictiveQuantBTCSystem:
    """
    🚀 Unified Predictive Quant BTC System
    
    A comprehensive cryptocurrency trading prediction system that combines:
    - Real-time data collection
    - Advanced feature engineering
    - XGBoost machine learning
    - Backtesting and performance analysis
    - Diagnostic visualizations
    - Automated improvement recommendations
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the unified system."""
        print("🚀 INITIALIZING PREDICTIVE QUANT BTC-XGBOOST SYSTEM")
        print("=" * 80)
        
        # Load configuration and setup logging
        self.config = get_config()
        self.logger = setup_logging()
        
        # Initialize components
        self.data_collector = BinanceDataCollector()
        self.feature_engine = FeatureEngine()
        self.visualizer = TradingVisualizer()
        self.diagnostic_visualizer = XGBoostDiagnosticVisualizer()
        self.scaler = RobustScaler()
        
        # Performance tracking
        self.start_time = time.time()
        self.performance_metrics = {}
        
        print("✅ System initialized successfully!")
        print()
    
    def collect_data(self, symbol: str = "BTC/USDT", 
                    timeframe: str = "1h", 
                    days: int = 60) -> pd.DataFrame:
        """
        📊 Collect real-time cryptocurrency data from Binance.
        
        Args:
            symbol: Trading pair (default: BTC/USDT)
            timeframe: Data timeframe (default: 1h)
            days: Number of days to collect (default: 60)
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"📊 STEP 1: COLLECTING {symbol} DATA")
        print("-" * 50)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Convert to strings for the collector
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"   📈 Symbol: {symbol}")
        print(f"   ⏰ Timeframe: {timeframe}")
        print(f"   📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   🎯 Target days: {days}")
        
        # Collect data
        try:
            data = self.data_collector.collect_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            # Ensure datetime index for time-based features
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif not isinstance(data.index, pd.DatetimeIndex):
                # Create a datetime index if none exists
                data.index = pd.date_range(start=start_date, periods=len(data), freq='1h')
            
            # Data quality metrics
            print(f"   ✅ Collected: {len(data)} candles ({len(data)/24:.1f} days)")
            print(f"   📈 Price range: ${data['close'].min():.0f} - ${data['close'].max():.0f}")
            print(f"   📊 Avg volume: {data['volume'].mean():.0f}")
            print(f"   💾 Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            print()
            
            return data
            
        except Exception as e:
            print(f"❌ Data collection failed: {e}")
            raise
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        🔧 Create comprehensive feature set with technical indicators.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 STEP 2: ADVANCED FEATURE ENGINEERING")
        print("-" * 50)
        
        print("   🧮 Creating comprehensive feature set...")
        
        try:
            # Generate all features
            features = self.feature_engine.create_all_features(data)
            
            # Feature statistics
            feature_count = len(features.columns)
            memory_usage = features.memory_usage(deep=True).sum() / 1024**2
            completeness = (1 - features.isnull().mean().mean()) * 100
            
            print(f"   ✅ Created: {feature_count} features")
            print(f"   💾 Memory usage: {memory_usage:.1f} MB")
            print(f"   📏 Data completeness: {completeness:.1f}%")
            print()
            
            return features
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            raise
    
    def create_target(self, data: pd.DataFrame, lookahead: int = 4) -> pd.Series:
        """
        🎯 Create target variable for price direction prediction.
        
        Args:
            data: OHLCV data
            lookahead: Hours to look ahead for target (default: 4)
            
        Returns:
            Binary target series (1=up, 0=down)
        """
        print("🎯 STEP 3: CREATING TARGET VARIABLE")
        print("-" * 50)
        
        # Calculate future returns
        future_returns = data['close'].pct_change(lookahead).shift(-lookahead)
        target = (future_returns > 0).astype(int)
        
        # Target statistics
        positive_rate = target.mean() * 100
        target_entropy = -np.sum([p * np.log2(p) for p in [target.mean(), 1-target.mean()] if p > 0])
        
        print(f"   ✅ Target: predicting {lookahead}-hour price direction")
        print(f"   📊 Positive rate: {positive_rate:.1f}%")
        print(f"   ⚖️ Class balance: {'Good' if 45 <= positive_rate <= 55 else 'Imbalanced'}")
        print(f"   🔍 Target entropy: {target_entropy:.3f}")
        print()
        
        return target
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple:
        """
        ✂️ Prepare data for machine learning with intelligent splitting.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("✂️ STEP 4: INTELLIGENT DATA PREPARATION")
        print("-" * 50)
        
        # Remove NaN values
        valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
        X_clean = features[valid_indices]
        y_clean = target[valid_indices]
        
        # Time-aware splitting (no data leakage)
        train_size = int(0.6 * len(X_clean))
        val_size = int(0.2 * len(X_clean))
        
        X_train = X_clean[:train_size]
        y_train = y_clean[:train_size]
        X_val = X_clean[train_size:train_size + val_size]
        y_val = y_clean[train_size:train_size + val_size]
        X_test = X_clean[train_size + val_size:]
        y_test = y_clean[train_size + val_size:]
        
        print(f"   📈 Training set: {len(X_train)} samples ({len(X_train)/len(X_clean)*100:.1f}%)")
        print(f"   🔍 Validation set: {len(X_val)} samples ({len(X_val)/len(X_clean)*100:.1f}%)")
        print(f"   📉 Test set: {len(X_test)} samples ({len(X_test)/len(X_clean)*100:.1f}%)")
        print(f"   ⚖️ Train class balance: {y_train.mean()*100:.1f}% positive")
        print()
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       top_k: int = 40) -> Tuple[pd.DataFrame, List[str]]:
        """
        🎯 Intelligent feature selection using multiple methods.
        
        Args:
            X_train: Training features
            y_train: Training target
            top_k: Number of features to select
            
        Returns:
            Tuple of (selected_features_df, feature_names)
        """
        print("🎯 STEP 5: INTELLIGENT FEATURE SELECTION")
        print("-" * 50)
        
        print("   🔍 Testing multiple feature selection approaches...")
        
        # Method 1: Mutual Information
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=top_k)
        X_mi = selector_mi.fit_transform(X_train, y_train)
        features_mi = X_train.columns[selector_mi.get_support()].tolist()
        print(f"   📊 Mutual Info selected: {len(features_mi)} features")
        
        # Method 2: F-score
        selector_f = SelectKBest(score_func=f_classif, k=top_k)
        X_f = selector_f.fit_transform(X_train, y_train)
        features_f = X_train.columns[selector_f.get_support()].tolist()
        print(f"   📈 F-score selected: {len(features_f)} features")
        
        # Method 3: Correlation with target
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        features_corr = correlations.head(top_k).index.tolist()
        print(f"   🔗 Correlation selected: {len(features_corr)} features")
        
        # Combine and select best
        all_selected = set(features_mi + features_f + features_corr)
        print(f"   🎯 Combined unique features: {len(all_selected)}")
        
        # Final selection using mutual information on combined set
        if len(all_selected) > top_k:
            X_combined = X_train[list(all_selected)]
            selector_final = SelectKBest(score_func=mutual_info_classif, k=top_k)
            selector_final.fit(X_combined, y_train)
            final_features = X_combined.columns[selector_final.get_support()].tolist()
        else:
            final_features = list(all_selected)
        
        X_selected = X_train[final_features]
        
        print(f"   ✅ Final selected features: {len(final_features)}")
        print(f"   🔝 Top 5 features: {final_features[:5]}")
        print()
        
        return X_selected, final_features
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
        """
        🤖 Train XGBoost model with optimal parameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained XGBoost model
        """
        print("🤖 STEP 6: TRAINING XGBOOST MODEL")
        print("-" * 50)
        
        print("   🏗️ Building and training model...")
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Configure XGBoost parameters
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
        
        # Train model
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        print(f"   ✅ Model trained with {model.n_estimators} trees")
        print(f"   🌳 Max depth: {model.max_depth}")
        print(f"   📚 Learning rate: {model.learning_rate}")
        print()
        
        return model
    
    def evaluate_model(self, model: xgb.XGBClassifier, 
                      X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series,
                      feature_names: List[str]) -> Dict:
        """
        📊 Comprehensive model evaluation and analysis.
        
        Args:
            model: Trained XGBoost model
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: List of feature names
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("📊 STEP 7: COMPREHENSIVE MODEL EVALUATION")
        print("-" * 50)
        
        # Scale test data
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        auc_score = roc_auc_score(y_test, y_test_proba)
        overfitting_gap = train_accuracy - test_accuracy
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Confidence analysis
        high_conf_mask = (y_test_proba > 0.7) | (y_test_proba < 0.3)
        medium_conf_mask = (y_test_proba >= 0.4) & (y_test_proba <= 0.6)
        low_conf_mask = ~(high_conf_mask | medium_conf_mask)
        
        high_conf_acc = accuracy_score(y_test[high_conf_mask], y_test_pred[high_conf_mask]) if high_conf_mask.sum() > 0 else 0
        medium_conf_acc = accuracy_score(y_test[medium_conf_mask], y_test_pred[medium_conf_mask]) if medium_conf_mask.sum() > 0 else 0
        low_conf_acc = accuracy_score(y_test[low_conf_mask], y_test_pred[low_conf_mask]) if low_conf_mask.sum() > 0 else 0
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'auc_score': auc_score,
            'overfitting_gap': overfitting_gap,
            'feature_importance': feature_importance,
            'y_test_proba': y_test_proba,
            'high_conf_samples': high_conf_mask.sum(),
            'medium_conf_samples': medium_conf_mask.sum(),
            'low_conf_samples': low_conf_mask.sum(),
            'high_conf_accuracy': high_conf_acc,
            'medium_conf_accuracy': medium_conf_acc,
            'low_conf_accuracy': low_conf_acc
        }
        
        print("   🧠 Model Performance Analysis:")
        print(f"      Training Accuracy: {train_accuracy:.1%}")
        print(f"      Test Accuracy: {test_accuracy:.1%}")
        print(f"      AUC Score: {auc_score:.3f}")
        print(f"      Overfitting Gap: {overfitting_gap:.3f}")
        print("   🎯 Confidence Analysis:")
        print(f"      High Confidence: {high_conf_mask.sum()} samples (Acc: {high_conf_acc:.1%})")
        print(f"      Medium Confidence: {medium_conf_mask.sum()} samples (Acc: {medium_conf_acc:.1%})")
        print(f"      Low Confidence: {low_conf_mask.sum()} samples (Acc: {low_conf_acc:.1%})")
        print("   🔍 Feature Insights:")
        print(f"      Most Important: {feature_importance.iloc[0]['feature']}")
        print(f"      Top 5 Concentration: {feature_importance.head(5)['importance'].sum():.1%}")
        print()
        
        return metrics
    
    def backtest_strategy(self, data: pd.DataFrame, 
                         y_test_proba: np.ndarray,
                         test_start_idx: int) -> Dict:
        """
        📈 Backtest trading strategy based on model predictions.
        
        Args:
            data: Original price data
            y_test_proba: Model probability predictions
            test_start_idx: Starting index for test period
            
        Returns:
            Dictionary with backtesting results
        """
        print("📈 STEP 8: BACKTESTING TRADING STRATEGY")
        print("-" * 50)
        
        # Align predictions with price data
        test_data = data.iloc[test_start_idx:test_start_idx + len(y_test_proba)]
        returns = test_data['close'].pct_change().dropna()
        
        # Ensure predictions and returns have same length
        min_length = min(len(y_test_proba), len(returns))
        y_test_proba_aligned = y_test_proba[:min_length]
        returns_aligned = returns.iloc[:min_length]
        
        # Create signals based on predictions
        signals = pd.Series((y_test_proba_aligned > 0.5).astype(int), index=returns_aligned.index)
        
        # Calculate strategy returns
        strategy_returns = returns_aligned * signals.shift(1).fillna(0)
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        buy_hold_return = (test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1
        volatility = strategy_returns.std() * np.sqrt(24 * 365)  # Annualized
        sharpe_ratio = (strategy_returns.mean() * 24 * 365) / (strategy_returns.std() * np.sqrt(24 * 365)) if strategy_returns.std() > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / rolling_max - 1)
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (strategy_returns > 0).mean()
        
        backtest_results = {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'strategy_returns': strategy_returns,
            'signals': signals
        }
        
        print(f"   💰 Strategy Performance:")
        print(f"      Total Return: {total_return:.1%}")
        print(f"      Buy & Hold Return: {buy_hold_return:.1%}")
        print(f"      Excess Return: {total_return - buy_hold_return:.1%}")
        print(f"      Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"      Max Drawdown: {max_drawdown:.1%}")
        print(f"      Win Rate: {win_rate:.1%}")
        print()
        
        return backtest_results
    
    def generate_recommendations(self, metrics: Dict) -> List[str]:
        """
        💡 Generate automated improvement recommendations.
        
        Args:
            metrics: Model evaluation metrics
            
        Returns:
            List of improvement recommendations
        """
        print("💡 STEP 9: GENERATING IMPROVEMENT RECOMMENDATIONS")
        print("-" * 50)
        
        recommendations = []
        
        # Check for overfitting
        if metrics['overfitting_gap'] > 0.1:
            recommendations.append({
                'priority': 'High',
                'category': 'Overfitting',
                'issue': f"High overfitting gap: {metrics['overfitting_gap']:.3f}",
                'recommendation': "Increase regularization (higher reg_alpha/reg_lambda) or reduce max_depth"
            })
        
        # Check model confidence
        if metrics['medium_conf_samples'] > metrics['high_conf_samples']:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Model Confidence',
                'issue': "Too many medium-confidence predictions",
                'recommendation': "Adjust decision threshold or use calibration techniques"
            })
        
        # Check for class imbalance issues
        if metrics['low_conf_accuracy'] < 0.4:
            recommendations.append({
                'priority': 'High',
                'category': 'Prediction Quality',
                'issue': "Low accuracy in low-confidence predictions",
                'recommendation': "Improve feature engineering or consider ensemble methods"
            })
        
        # Feature concentration
        top_5_importance = metrics['feature_importance'].head(5)['importance'].sum()
        if top_5_importance > 0.8:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Feature Diversity',
                'issue': f"High feature concentration: {top_5_importance:.1%}",
                'recommendation': "Diversify feature set or use feature selection to reduce dependency"
            })
        
        print("   🎯 System Improvement Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. [{rec['priority']} Priority] {rec['category']}")
            print(f"      Issue: {rec['issue']}")
            print(f"      Recommendation: {rec['recommendation']}")
            print()
        
        return recommendations
    
    def create_visualizations(self, data: pd.DataFrame, metrics: Dict,
                            backtest_results: Dict, model: xgb.XGBClassifier,
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            feature_names: List[str]) -> None:
        """
        🎨 Create comprehensive visualization dashboards.
        
        Args:
            data: Original price data
            metrics: Model evaluation metrics
            backtest_results: Backtesting results
            model: Trained model
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: Feature names
        """
        print("🎨 STEP 10: CREATING COMPREHENSIVE VISUALIZATIONS")
        print("-" * 50)
        
        try:
            # 1. Performance Dashboard
            print("   📊 Creating performance dashboard...")
            self.visualizer.create_performance_dashboard(
                returns=backtest_results['strategy_returns'],
                signals=backtest_results['signals'],
                predictions=metrics['y_test_proba'],
                accuracy=metrics['test_accuracy'],
                sharpe_ratio=backtest_results['sharpe_ratio'],
                max_drawdown=backtest_results['max_drawdown'],
                save_path='outputs/visualizations/performance_dashboard.png'
            )
            
            # 2. Feature Analysis
            print("   🔍 Creating feature analysis...")
            correlation_matrix = X_train[feature_names[:15]].corr() if len(feature_names) >= 15 else X_train.corr()
            self.visualizer.create_feature_analysis(
                feature_importance=metrics['feature_importance'],
                correlation_matrix=correlation_matrix,
                save_path='outputs/visualizations/feature_analysis.png'
            )
            
            # 3. Technical Analysis
            print("   📈 Creating technical analysis...")
            self.visualizer.create_technical_analysis(
                data=data,
                signals=backtest_results['signals'],
                predictions=metrics['y_test_proba'],
                save_path='outputs/visualizations/technical_analysis.png'
            )
            
            # 4. Comprehensive Diagnostic Dashboard
            print("   🔬 Creating diagnostic dashboard...")
            self.diagnostic_visualizer.create_comprehensive_dashboard(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                y_proba_test=metrics['y_test_proba'],
                feature_names=feature_names,
                save_path='outputs/visualizations/diagnostic_dashboard.png'
            )
            
            # 5. Feature Deep Dive
            print("   🔍 Creating feature deep dive...")
            self.diagnostic_visualizer.create_feature_deep_dive(
                model=model,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                save_path='outputs/visualizations/feature_deep_dive.png'
            )
            
            print("   ✅ All visualizations created successfully!")
            print()
            
        except Exception as e:
            print(f"❌ Visualization creation failed: {e}")
            print(f"   ❌ Visualization error: {e}")
    
    def run_complete_analysis(self) -> None:
        """
        🚀 Run the complete end-to-end analysis pipeline.
        """
        try:
            print("🚀 STARTING COMPLETE PREDICTIVE QUANT BTC ANALYSIS")
            print("=" * 80)
            print("This system provides comprehensive cryptocurrency trading prediction")
            print("using advanced XGBoost machine learning with diagnostic insights.")
            print("=" * 80)
            print()
            
            # Step 1: Data Collection
            data = self.collect_data()
            
            # Step 2: Feature Engineering
            features = self.engineer_features(data)
            
            # Step 3: Target Creation
            target = self.create_target(data)
            
            # Step 4: Data Preparation
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(features, target)
            
            # Step 5: Feature Selection
            X_train_selected, feature_names = self.select_features(X_train, y_train)
            X_val_selected = X_val[feature_names]
            X_test_selected = X_test[feature_names]
            
            # Step 6: Model Training
            model = self.train_model(X_train_selected, y_train, X_val_selected, y_val)
            
            # Step 7: Model Evaluation
            metrics = self.evaluate_model(model, X_train_selected, y_train, X_test_selected, y_test, feature_names)
            
            # Step 8: Backtesting
            test_start_idx = len(X_train) + len(X_val)
            backtest_results = self.backtest_strategy(data, metrics['y_test_proba'], test_start_idx)
            
            # Step 9: Recommendations
            recommendations = self.generate_recommendations(metrics)
            
            # Step 10: Visualizations
            self.create_visualizations(
                data, metrics, backtest_results, model,
                X_train_selected, y_train, X_test_selected, y_test, feature_names
            )
            
            # Final Summary
            execution_time = time.time() - self.start_time
            
            print("=" * 80)
            print("🎉 COMPLETE ANALYSIS FINISHED SUCCESSFULLY!")
            print("=" * 80)
            print()
            print("📊 FINAL SUMMARY:")
            print(f"   🎯 Model Accuracy: {metrics['test_accuracy']:.1%}")
            print(f"   📈 Strategy Return: {backtest_results['total_return']:.1%}")
            print(f"   📊 Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"   🔍 Features Used: {len(feature_names)}")
            print(f"   💡 Recommendations: {len(recommendations)}")
            print(f"   ⚡ Execution Time: {execution_time:.1f} seconds")
            print()
            print("🎨 GENERATED FILES:")
            print("   📊 outputs/visualizations/performance_dashboard.png - Trading performance analysis")
            print("   🔍 outputs/visualizations/feature_analysis.png - Feature importance and correlations")
            print("   📈 outputs/visualizations/technical_analysis.png - Technical indicators and signals")
            print("   🔬 outputs/visualizations/diagnostic_dashboard.png - Model diagnostic insights")
            print("   🔍 outputs/visualizations/feature_deep_dive.png - Advanced feature analysis")
            print()
            print("🚀 Use these insights to optimize your trading strategy!")
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Analysis pipeline failed: {e}")
            raise


def main():
    """🚀 Main entry point for the Predictive Quant BTC System."""
    try:
        # Create and run the unified system
        system = PredictiveQuantBTCSystem()
        system.run_complete_analysis()
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()