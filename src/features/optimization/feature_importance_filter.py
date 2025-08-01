"""
Feature Importance Filter for Smart Indicator Selection
=====================================================

Uses ensemble machine learning models to identify the most predictive technical
indicators before parameter optimization. This dramatically reduces the search
space and focuses optimization on indicators that actually matter.

Features:
- Multiple ensemble models (Random Forest, XGBoost, LightGBM)
- Feature importance ranking and selection
- Correlation analysis and redundancy removal
- Stability testing across time periods
- Automatic feature preprocessing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Ensemble model imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, r2_score

# Optional advanced models (install if available)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from ..technical_indicators import TechnicalIndicators
from ..feature_engine import FeatureEngine
from ...utils.logging import get_logger


@dataclass
class FeatureImportanceResult:
    """Results from feature importance analysis"""
    feature_scores: Dict[str, float]
    selected_features: List[str]
    redundant_features: List[str]
    feature_categories: Dict[str, List[str]]
    model_performance: Dict[str, float]
    stability_scores: Dict[str, float]


@dataclass
class FeatureFilterConfig:
    """Configuration for feature importance filtering"""
    max_features: int = 50
    min_importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    stability_threshold: float = 0.7
    cross_validation_folds: int = 5
    use_advanced_models: bool = True


class FeatureImportanceFilter:
    """
    Advanced feature importance analyzer using ensemble methods.
    
    Identifies the most predictive technical indicators and removes
    redundant features to optimize the parameter search process.
    """
    
    def __init__(self, config: FeatureFilterConfig = None):
        """
        Initialize feature importance filter.
        
        Args:
            config: Configuration for feature filtering
        """
        self.config = config or FeatureFilterConfig()
        self.logger = get_logger().get_logger()
        
        # Feature engineering components
        self.technical_indicators = TechnicalIndicators()
        self.feature_engine = FeatureEngine()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model storage
        self.trained_models = {}
        self.feature_importances = {}
        self.selected_features = []
        
    def analyze_feature_importance(self,
                                 data: pd.DataFrame,
                                 target_column: str = None,
                                 prediction_horizon: int = 1) -> FeatureImportanceResult:
        """
        Comprehensive feature importance analysis.
        
        Args:
            data: Historical OHLCV data
            target_column: Target variable name (if None, creates price direction target)
            prediction_horizon: Hours ahead to predict
            
        Returns:
            FeatureImportanceResult with comprehensive analysis
        """
        self.logger.info("Starting comprehensive feature importance analysis")
        
        # Generate comprehensive features
        features = self.feature_engine.create_all_features(data)
        
        # Create target variable if not provided
        if target_column is None:
            target = self._create_price_direction_target(data, prediction_horizon)
            target_name = f'price_direction_{prediction_horizon}h'
        else:
            target = data[target_column]
            target_name = target_column
        
        # Align features and target
        features, target = self._align_features_target(features, target)
        
        if len(features) == 0 or len(target) == 0:
            raise ValueError("No valid features or target data available")
        
        # Run multiple importance analysis methods
        importance_results = self._run_multiple_importance_analyses(features, target, target_name)
        
        # Combine importance scores
        combined_scores = self._combine_importance_scores(importance_results)
        
        # Select best features
        selected_features = self._select_best_features(
            features, combined_scores, target
        )
        
        # Remove redundant features
        final_features, redundant_features = self._remove_redundant_features(
            features[selected_features]
        )
        
        # Categorize features
        feature_categories = self._categorize_features(final_features)
        
        # Calculate stability scores
        stability_scores = self._calculate_feature_stability(
            features[final_features], target
        )
        
        # Calculate model performance
        model_performance = self._evaluate_feature_subset_performance(
            features[final_features], target
        )
        
        result = FeatureImportanceResult(
            feature_scores=combined_scores,
            selected_features=final_features,
            redundant_features=redundant_features,
            feature_categories=feature_categories,
            model_performance=model_performance,
            stability_scores=stability_scores
        )
        
        self.logger.info(f"Feature importance analysis completed:")
        self.logger.info(f"Selected {len(final_features)} features from {len(features.columns)} total")
        self.logger.info(f"Removed {len(redundant_features)} redundant features")
        
        return result
    
    def filter_indicator_parameters(self,
                                  data: pd.DataFrame,
                                  all_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter indicator parameters based on feature importance.
        
        Args:
            data: Historical OHLCV data
            all_parameters: All possible indicator parameters
            
        Returns:
            Filtered parameters focusing on important indicators
        """
        self.logger.info("Filtering indicator parameters based on feature importance")
        
        # Analyze feature importance
        importance_result = self.analyze_feature_importance(data)
        
        # Map features to indicator types
        indicator_importance = self._map_features_to_indicators(
            importance_result.feature_scores
        )
        
        # Filter parameters based on indicator importance
        filtered_parameters = {}
        importance_threshold = self.config.min_importance_threshold
        
        for indicator, importance in indicator_importance.items():
            if importance >= importance_threshold:
                # Include parameters for this indicator
                indicator_params = self._get_indicator_parameters(indicator, all_parameters)
                filtered_parameters.update(indicator_params)
                
                self.logger.info(f"Including {indicator} (importance: {importance:.4f})")
            else:
                self.logger.info(f"Excluding {indicator} (importance: {importance:.4f})")
        
        self.logger.info(f"Filtered parameters: {len(filtered_parameters)} from {len(all_parameters)}")
        return filtered_parameters
    
    def get_indicator_rankings(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Get ranking of technical indicators by predictive power.
        
        Args:
            data: Historical OHLCV data
            
        Returns:
            Dictionary mapping indicator names to importance scores
        """
        importance_result = self.analyze_feature_importance(data)
        return self._map_features_to_indicators(importance_result.feature_scores)
    
    def _create_price_direction_target(self,
                                     data: pd.DataFrame,
                                     horizon: int) -> pd.Series:
        """Create price direction target (binary classification)"""
        try:
            # Calculate future returns
            future_returns = data['close'].pct_change(periods=horizon).shift(-horizon)
            
            # Create binary target (1 = price goes up, 0 = price goes down)
            target = (future_returns > 0).astype(int)
            
            # Remove NaN values
            target = target.dropna()
            
            return target
            
        except Exception as e:
            self.logger.error(f"Error creating price direction target: {e}")
            return pd.Series(dtype=int)
    
    def _align_features_target(self,
                             features: pd.DataFrame,
                             target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target data"""
        try:
            # Find common index
            common_index = features.index.intersection(target.index)
            
            if len(common_index) == 0:
                self.logger.warning("No common index between features and target")
                return pd.DataFrame(), pd.Series(dtype=float)
            
            # Align data
            aligned_features = features.loc[common_index]
            aligned_target = target.loc[common_index]
            
            # Remove rows with NaN in target
            valid_mask = ~aligned_target.isna()
            aligned_features = aligned_features[valid_mask]
            aligned_target = aligned_target[valid_mask]
            
            # Remove features with too many NaNs (>50%)
            nan_threshold = 0.5
            valid_features = aligned_features.columns[
                aligned_features.isna().mean() < nan_threshold
            ]
            aligned_features = aligned_features[valid_features]
            
            # Fill remaining NaNs
            aligned_features = aligned_features.ffill().fillna(0)
            
            self.logger.info(f"Aligned data: {len(aligned_features)} samples, {len(aligned_features.columns)} features")
            
            return aligned_features, aligned_target
            
        except Exception as e:
            self.logger.error(f"Error aligning features and target: {e}")
            return pd.DataFrame(), pd.Series(dtype=float)
    
    def _run_multiple_importance_analyses(self,
                                        features: pd.DataFrame,
                                        target: pd.Series,
                                        target_name: str) -> Dict[str, Dict[str, float]]:
        """Run multiple feature importance analysis methods"""
        importance_results = {}
        
        # 1. Random Forest Importance
        try:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(features, target)
            rf_importance = dict(zip(features.columns, rf_model.feature_importances_))
            importance_results['random_forest'] = rf_importance
            self.trained_models['random_forest'] = rf_model
            
            self.logger.info("Random Forest importance analysis completed")
            
        except Exception as e:
            self.logger.warning(f"Random Forest analysis failed: {e}")
        
        # 2. Extra Trees Importance
        try:
            et_model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            et_model.fit(features, target)
            et_importance = dict(zip(features.columns, et_model.feature_importances_))
            importance_results['extra_trees'] = et_importance
            self.trained_models['extra_trees'] = et_model
            
            self.logger.info("Extra Trees importance analysis completed")
            
        except Exception as e:
            self.logger.warning(f"Extra Trees analysis failed: {e}")
        
        # 3. XGBoost Importance (if available)
        if HAS_XGBOOST and self.config.use_advanced_models:
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                xgb_model.fit(features, target)
                xgb_importance = dict(zip(features.columns, xgb_model.feature_importances_))
                importance_results['xgboost'] = xgb_importance
                self.trained_models['xgboost'] = xgb_model
                
                self.logger.info("XGBoost importance analysis completed")
                
            except Exception as e:
                self.logger.warning(f"XGBoost analysis failed: {e}")
        
        # 4. Mutual Information
        try:
            mi_scores = mutual_info_classif(features, target, random_state=42)
            mi_importance = dict(zip(features.columns, mi_scores))
            importance_results['mutual_info'] = mi_importance
            
            self.logger.info("Mutual Information analysis completed")
            
        except Exception as e:
            self.logger.warning(f"Mutual Information analysis failed: {e}")
        
        # 5. F-statistic
        try:
            f_scores, _ = f_classif(features, target)
            # Normalize F-scores
            f_scores = f_scores / f_scores.max() if f_scores.max() > 0 else f_scores
            f_importance = dict(zip(features.columns, f_scores))
            importance_results['f_statistic'] = f_importance
            
            self.logger.info("F-statistic analysis completed")
            
        except Exception as e:
            self.logger.warning(f"F-statistic analysis failed: {e}")
        
        return importance_results
    
    def _combine_importance_scores(self,
                                 importance_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Combine importance scores from multiple methods"""
        if not importance_results:
            return {}
        
        # Get all features
        all_features = set()
        for method_scores in importance_results.values():
            all_features.update(method_scores.keys())
        
        # Combine scores using ensemble average
        combined_scores = {}
        
        for feature in all_features:
            scores = []
            for method, method_scores in importance_results.items():
                if feature in method_scores:
                    # Normalize score (0-1 range)
                    max_score = max(method_scores.values()) if method_scores.values() else 1.0
                    normalized_score = method_scores[feature] / max_score if max_score > 0 else 0.0
                    scores.append(normalized_score)
            
            # Average score across methods
            combined_scores[feature] = np.mean(scores) if scores else 0.0
        
        # Normalize final scores
        max_combined = max(combined_scores.values()) if combined_scores.values() else 1.0
        if max_combined > 0:
            combined_scores = {k: v/max_combined for k, v in combined_scores.items()}
        
        return combined_scores
    
    def _select_best_features(self,
                            features: pd.DataFrame,
                            importance_scores: Dict[str, float],
                            target: pd.Series) -> List[str]:
        """Select best features based on importance scores"""
        # Sort features by importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top features
        max_features = min(self.config.max_features, len(sorted_features))
        min_threshold = self.config.min_importance_threshold
        
        selected_features = []
        for feature, score in sorted_features[:max_features]:
            if score >= min_threshold:
                selected_features.append(feature)
            else:
                break
        
        # Ensure we have at least some features
        if len(selected_features) == 0 and sorted_features:
            # Take top 10 features regardless of threshold
            selected_features = [f[0] for f in sorted_features[:10]]
        
        return selected_features
    
    def _remove_redundant_features(self,
                                 features: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Remove highly correlated (redundant) features"""
        if len(features.columns) <= 1:
            return list(features.columns), []
        
        # Calculate correlation matrix
        correlation_matrix = features.corr().abs()
        
        # Find highly correlated feature pairs
        redundant_features = set()
        feature_list = list(features.columns)
        
        for i in range(len(feature_list)):
            for j in range(i + 1, len(feature_list)):
                feature1, feature2 = feature_list[i], feature_list[j]
                
                if correlation_matrix.loc[feature1, feature2] > self.config.correlation_threshold:
                    # Keep the feature with higher variance (more informative)
                    var1 = features[feature1].var()
                    var2 = features[feature2].var()
                    
                    if var1 >= var2:
                        redundant_features.add(feature2)
                    else:
                        redundant_features.add(feature1)
        
        # Final feature list
        final_features = [f for f in features.columns if f not in redundant_features]
        redundant_list = list(redundant_features)
        
        self.logger.info(f"Removed {len(redundant_list)} redundant features")
        return final_features, redundant_list
    
    def _categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type"""
        categories = {
            'price': [],
            'volume': [],
            'volatility': [],
            'momentum': [],
            'trend': [],
            'oscillators': [],
            'time': [],
            'other': []
        }
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['price', 'close', 'open', 'high', 'low', 'return']):
                categories['price'].append(feature)
            elif any(keyword in feature_lower for keyword in ['volume', 'obv', 'vwap']):
                categories['volume'].append(feature)
            elif any(keyword in feature_lower for keyword in ['volatility', 'atr', 'std', 'bb_']):
                categories['volatility'].append(feature)
            elif any(keyword in feature_lower for keyword in ['rsi', 'stoch', 'williams', 'cci']):
                categories['oscillators'].append(feature)
            elif any(keyword in feature_lower for keyword in ['macd', 'momentum', 'roc']):
                categories['momentum'].append(feature)
            elif any(keyword in feature_lower for keyword in ['sma', 'ema', 'ma_', 'trend']):
                categories['trend'].append(feature)
            elif any(keyword in feature_lower for keyword in ['hour', 'day', 'month', 'session']):
                categories['time'].append(feature)
            else:
                categories['other'].append(feature)
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        return categories
    
    def _calculate_feature_stability(self,
                                   features: pd.DataFrame,
                                   target: pd.Series) -> Dict[str, float]:
        """Calculate stability of feature importance across time periods"""
        stability_scores = {}
        
        if len(features) < 100:  # Need sufficient data for stability analysis
            return {feature: 1.0 for feature in features.columns}
        
        try:
            # Split data into time-based folds
            tscv = TimeSeriesSplit(n_splits=min(5, len(features) // 50))
            
            # Calculate importance for each feature across folds
            feature_importance_across_folds = {feature: [] for feature in features.columns}
            
            for train_idx, test_idx in tscv.split(features):
                X_train = features.iloc[train_idx]
                y_train = target.iloc[train_idx]
                
                if len(X_train) < 20 or len(y_train.unique()) < 2:
                    continue
                
                # Train simple model for importance
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                
                # Store importance for each feature
                for feature, importance in zip(features.columns, rf.feature_importances_):
                    feature_importance_across_folds[feature].append(importance)
            
            # Calculate stability (1 - coefficient of variation)
            for feature, importances in feature_importance_across_folds.items():
                if len(importances) > 1:
                    mean_importance = np.mean(importances)
                    std_importance = np.std(importances)
                    
                    if mean_importance > 0:
                        cv = std_importance / mean_importance
                        stability = 1.0 / (1.0 + cv)  # Higher stability = lower variation
                    else:
                        stability = 0.0
                    
                    stability_scores[feature] = stability
                else:
                    stability_scores[feature] = 1.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating feature stability: {e}")
            # Return default stability scores
            stability_scores = {feature: 0.5 for feature in features.columns}
        
        return stability_scores
    
    def _evaluate_feature_subset_performance(self,
                                           features: pd.DataFrame,
                                           target: pd.Series) -> Dict[str, float]:
        """Evaluate performance of selected feature subset"""
        performance = {}
        
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
            
            # Random Forest performance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_scores = cross_val_score(rf, features, target, cv=tscv, scoring='accuracy')
            performance['random_forest_accuracy'] = rf_scores.mean()
            performance['random_forest_std'] = rf_scores.std()
            
            # XGBoost performance (if available)
            if HAS_XGBOOST:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1
                )
                xgb_scores = cross_val_score(xgb_model, features, target, cv=tscv, scoring='accuracy')
                performance['xgboost_accuracy'] = xgb_scores.mean()
                performance['xgboost_std'] = xgb_scores.std()
            
            self.logger.info(f"Feature subset performance - RF: {performance['random_forest_accuracy']:.4f}")
            
        except Exception as e:
            self.logger.warning(f"Error evaluating feature subset performance: {e}")
            performance = {'random_forest_accuracy': 0.5, 'random_forest_std': 0.0}
        
        return performance
    
    def _map_features_to_indicators(self, feature_scores: Dict[str, float]) -> Dict[str, float]:
        """Map feature scores to indicator types"""
        indicator_scores = {}
        
        # Define indicator mapping
        indicator_mapping = {
            'RSI': ['rsi'],
            'MACD': ['macd'],
            'Bollinger_Bands': ['bb_', 'bollinger'],
            'Moving_Averages': ['sma_', 'ema_', 'ma_'],
            'Stochastic': ['stoch'],
            'Williams_R': ['williams'],
            'CCI': ['cci'],
            'ATR': ['atr'],
            'Momentum': ['momentum', 'roc'],
            'Volume': ['volume', 'obv', 'vwap'],
            'Volatility': ['volatility', 'garch'],
            'Price': ['price', 'return', 'close', 'open'],
            'ADX': ['adx', 'plus_di', 'minus_di'],
            'Ichimoku': ['tenkan', 'kijun', 'senkou', 'chikou']
        }
        
        # Calculate scores for each indicator
        for indicator, keywords in indicator_mapping.items():
            scores = []
            for feature, score in feature_scores.items():
                feature_lower = feature.lower()
                if any(keyword in feature_lower for keyword in keywords):
                    scores.append(score)
            
            # Average score for this indicator type
            indicator_scores[indicator] = np.mean(scores) if scores else 0.0
        
        return indicator_scores
    
    def _get_indicator_parameters(self,
                                indicator: str,
                                all_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for specific indicator type"""
        indicator_params = {}
        
        # Map indicator names to parameter prefixes
        param_mapping = {
            'RSI': 'rsi_',
            'MACD': 'macd_',
            'Bollinger_Bands': 'bb_',
            'Moving_Averages': 'ma_',
            'Stochastic': 'stoch_',
            'ATR': 'atr_',
            'Williams_R': 'williams_',
            'CCI': 'cci_',
            'Momentum': 'momentum_',
            'ADX': 'adx_'
        }
        
        # Get relevant parameters
        prefix = param_mapping.get(indicator, indicator.lower() + '_')
        
        for param_name, param_value in all_parameters.items():
            if param_name.startswith(prefix):
                indicator_params[param_name] = param_value
        
        return indicator_params