"""
Advanced visualization module for XGBoost cryptocurrency trading system.
Creates comprehensive charts and dashboards for performance analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for professional-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TradingVisualizer:
    """
    Comprehensive visualization suite for trading system analysis.
    Creates professional charts for performance, risk, and technical analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 300):
        """
        Initialize the trading visualizer.
        
        Args:
            figsize: Default figure size for matplotlib charts
            dpi: Resolution for saved charts
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'danger': '#C73E1D',
            'info': '#31393C',
            'profit': '#00C853',
            'loss': '#FF1744',
            'neutral': '#757575'
        }
    
    def create_performance_dashboard(self, 
                                   results: Dict[str, Any],
                                   model_metrics: Dict[str, float],
                                   backtest_results: Dict[str, float],
                                   save_path: str = 'outputs/visualizations/trading_dashboard.png') -> None:
        """
        Create comprehensive performance dashboard.
        
        Args:
            results: Model training results
            model_metrics: Model performance metrics
            backtest_results: Backtesting performance results
            save_path: Path to save the dashboard
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout for dashboard
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model Performance Metrics (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_model_metrics(ax1, model_metrics)
        
        # 2. Feature Importance (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'feature_importance' in results:
            self._plot_feature_importance(ax2, results['feature_importance'])
        
        # 3. Cumulative Returns (Middle Left)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'cumulative_returns' in backtest_results:
            self._plot_cumulative_returns(ax3, backtest_results)
        
        # 4. Risk Metrics (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_risk_metrics(ax4, backtest_results)
        
        # 5. Prediction Confidence Distribution (Bottom Left)
        ax5 = fig.add_subplot(gs[2, :2])
        if 'prediction_probabilities' in results:
            self._plot_prediction_confidence(ax5, results['prediction_probabilities'])
        
        # 6. Confusion Matrix (Bottom Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        if 'y_true' in results and 'y_pred' in results:
            self._plot_confusion_matrix(ax6, results['y_true'], results['y_pred'])
        
        # 7. Performance Summary Table (Bottom)
        ax7 = fig.add_subplot(gs[3, :])
        self._create_performance_table(ax7, model_metrics, backtest_results)
        
        plt.suptitle('🚀 XGBoost Cryptocurrency Trading System - Performance Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 Performance dashboard saved as '{save_path}'")
    
    def create_strategy_analysis(self,
                               strategy_returns: pd.Series,
                               benchmark_returns: pd.Series,
                               signals: pd.Series,
                               prices: pd.Series,
                               save_path: str = 'outputs/visualizations/strategy_analysis.png') -> None:
        """
        Create detailed strategy analysis charts.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark (buy & hold) returns
            signals: Trading signals
            prices: Price series
            save_path: Path to save the analysis
        """
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # 1. Cumulative Returns Comparison
        strategy_cumret = (1 + strategy_returns).cumprod()
        benchmark_cumret = (1 + benchmark_returns).cumprod()
        
        axes[0, 0].plot(strategy_cumret.index, strategy_cumret.values, 
                       label='Strategy', color=self.colors['primary'], linewidth=2)
        axes[0, 0].plot(benchmark_cumret.index, benchmark_cumret.values,
                       label='Buy & Hold', color=self.colors['danger'], linewidth=2)
        axes[0, 0].set_title('📈 Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe Ratio
        rolling_window = min(30, len(strategy_returns) // 4)
        strategy_sharpe = strategy_returns.rolling(rolling_window).mean() / strategy_returns.rolling(rolling_window).std() * np.sqrt(252)
        benchmark_sharpe = benchmark_returns.rolling(rolling_window).mean() / benchmark_returns.rolling(rolling_window).std() * np.sqrt(252)
        
        axes[0, 1].plot(strategy_sharpe.index, strategy_sharpe.values, 
                       label='Strategy Sharpe', color=self.colors['success'], linewidth=2)
        axes[0, 1].plot(benchmark_sharpe.index, benchmark_sharpe.values,
                       label='Benchmark Sharpe', color=self.colors['neutral'], linewidth=2)
        axes[0, 1].set_title(f'⚡ Rolling Sharpe Ratio ({rolling_window}d)', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Drawdown Analysis
        strategy_dd = self._calculate_drawdown(strategy_cumret)
        benchmark_dd = self._calculate_drawdown(benchmark_cumret)
        
        axes[1, 0].fill_between(strategy_dd.index, strategy_dd.values, 0, 
                               alpha=0.7, color=self.colors['danger'], label='Strategy DD')
        axes[1, 0].fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                               alpha=0.4, color=self.colors['neutral'], label='Benchmark DD')
        axes[1, 0].set_title('📉 Drawdown Analysis', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        if len(monthly_returns) > 12:
            monthly_returns_matrix = self._create_monthly_returns_matrix(monthly_returns)
            im = axes[1, 1].imshow(monthly_returns_matrix, cmap='RdYlGn', aspect='auto')
            axes[1, 1].set_title('🗓️ Monthly Returns Heatmap', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=axes[1, 1])
        
        # 5. Price Chart with Signals
        axes[2, 0].plot(prices.index, prices.values, color=self.colors['info'], alpha=0.7, linewidth=1)
        
        # Mark buy signals
        buy_signals = signals[signals == 1]
        if len(buy_signals) > 0:
            axes[2, 0].scatter(buy_signals.index, prices.loc[buy_signals.index], 
                              color=self.colors['profit'], marker='^', s=50, label='Buy Signals', alpha=0.8)
        
        axes[2, 0].set_title('📊 Price Action with Trading Signals', fontsize=14, fontweight='bold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Return Distribution
        axes[2, 1].hist(strategy_returns.dropna(), bins=50, alpha=0.7, 
                       color=self.colors['primary'], density=True, label='Strategy')
        axes[2, 1].hist(benchmark_returns.dropna(), bins=50, alpha=0.5,
                       color=self.colors['danger'], density=True, label='Benchmark')
        axes[2, 1].set_title('📊 Return Distribution', fontsize=14, fontweight='bold')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.suptitle('📈 Trading Strategy Performance Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"📈 Strategy analysis saved as '{save_path}'")
    
    def create_feature_analysis(self,
                              feature_importance: pd.DataFrame,
                              correlation_matrix: Optional[pd.DataFrame] = None,
                              feature_evolution: Optional[pd.DataFrame] = None,
                              save_path: str = 'outputs/visualizations/feature_analysis.png') -> None:
        """
        Create comprehensive feature analysis charts.
        
        Args:
            feature_importance: Feature importance DataFrame
            correlation_matrix: Feature correlation matrix
            feature_evolution: Feature values over time
            save_path: Path to save the analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top Feature Importance
        top_features = feature_importance.head(15)
        bars = axes[0, 0].barh(range(len(top_features)), top_features['importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_title('🏆 Top 15 Feature Importance', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Color bars by importance
        for i, bar in enumerate(bars):
            if i < 5:
                bar.set_color(self.colors['success'])
            elif i < 10:
                bar.set_color(self.colors['primary'])
            else:
                bar.set_color(self.colors['neutral'])
        
        # 2. Feature Importance Distribution
        axes[0, 1].hist(feature_importance['importance'], bins=30, alpha=0.7, 
                       color=self.colors['secondary'], edgecolor='black')
        axes[0, 1].axvline(feature_importance['importance'].mean(), 
                          color=self.colors['danger'], linestyle='--', 
                          label=f'Mean: {feature_importance["importance"].mean():.4f}')
        axes[0, 1].set_title('📊 Feature Importance Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Category Analysis
        self._plot_feature_categories(axes[1, 0], feature_importance)
        
        # 4. Correlation Heatmap (if provided)
        if correlation_matrix is not None:
            top_features_corr = correlation_matrix.loc[top_features['feature'][:10], 
                                                     top_features['feature'][:10]]
            im = axes[1, 1].imshow(top_features_corr, cmap='coolwarm', aspect='auto')
            axes[1, 1].set_xticks(range(len(top_features_corr.columns)))
            axes[1, 1].set_yticks(range(len(top_features_corr.index)))
            axes[1, 1].set_xticklabels(top_features_corr.columns, rotation=45)
            axes[1, 1].set_yticklabels(top_features_corr.index)
            axes[1, 1].set_title('🔗 Top Features Correlation', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Correlation Matrix\nNot Available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('🔗 Feature Correlation', fontsize=14, fontweight='bold')
        
        plt.suptitle('🔍 Feature Importance & Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"🔍 Feature analysis saved as '{save_path}'")
    
    def create_technical_analysis_chart(self,
                                      data: pd.DataFrame,
                                      indicators: Dict[str, pd.Series],
                                      signals: pd.Series,
                                      save_path: str = 'outputs/visualizations/technical_analysis.png') -> None:
        """
        Create technical analysis chart with indicators and signals.
        
        Args:
            data: OHLCV data
            indicators: Dictionary of technical indicators
            signals: Trading signals
            save_path: Path to save the chart
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), 
                                gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # 1. Price Chart with Moving Averages
        axes[0].plot(data.index, data['close'], color=self.colors['info'], 
                    linewidth=1.5, label='Close Price')
        
        # Add moving averages if available
        ma_colors = [self.colors['primary'], self.colors['secondary'], self.colors['success']]
        ma_count = 0
        for name, indicator in indicators.items():
            if 'SMA' in name or 'EMA' in name:
                if ma_count < 3:  # Limit to 3 MAs for clarity
                    axes[0].plot(indicator.index, indicator.values, 
                               color=ma_colors[ma_count], alpha=0.8, label=name)
                    ma_count += 1
        
        # Add Bollinger Bands if available
        if 'BB_Upper' in indicators and 'BB_Lower' in indicators:
            axes[0].fill_between(indicators['BB_Upper'].index,
                                indicators['BB_Upper'].values,
                                indicators['BB_Lower'].values,
                                alpha=0.2, color=self.colors['neutral'], label='Bollinger Bands')
        
        # Mark trading signals
        buy_signals = signals[signals == 1]
        if len(buy_signals) > 0:
            axes[0].scatter(buy_signals.index, data.loc[buy_signals.index, 'close'],
                           color=self.colors['profit'], marker='^', s=60, 
                           label='Buy Signals', zorder=5)
        
        axes[0].set_title('📊 Price Chart with Technical Indicators', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Volume
        volume_colors = ['red' if data['close'].iloc[i] < data['open'].iloc[i] else 'green' 
                        for i in range(len(data))]
        axes[1].bar(data.index, data['volume'], color=volume_colors, alpha=0.6)
        axes[1].set_title('📊 Volume', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. RSI
        if 'RSI_14' in indicators:
            axes[2].plot(indicators['RSI_14'].index, indicators['RSI_14'].values,
                        color=self.colors['secondary'], linewidth=2)
            axes[2].axhline(y=70, color=self.colors['danger'], linestyle='--', alpha=0.7)
            axes[2].axhline(y=30, color=self.colors['profit'], linestyle='--', alpha=0.7)
            axes[2].fill_between(indicators['RSI_14'].index, 70, 100, alpha=0.2, color=self.colors['danger'])
            axes[2].fill_between(indicators['RSI_14'].index, 0, 30, alpha=0.2, color=self.colors['profit'])
        axes[2].set_title('📈 RSI (14)', fontsize=12, fontweight='bold')
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, alpha=0.3)
        
        # 4. MACD
        if 'MACD' in indicators and 'Signal' in indicators:
            axes[3].plot(indicators['MACD'].index, indicators['MACD'].values,
                        color=self.colors['primary'], linewidth=2, label='MACD')
            axes[3].plot(indicators['Signal'].index, indicators['Signal'].values,
                        color=self.colors['danger'], linewidth=2, label='Signal')
            if 'Histogram' in indicators:
                axes[3].bar(indicators['Histogram'].index, indicators['Histogram'].values,
                           alpha=0.6, color=self.colors['neutral'], label='Histogram')
            axes[3].legend()
        axes[3].set_title('📊 MACD', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle('📈 Technical Analysis Chart', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"📊 Technical analysis chart saved as '{save_path}'")
    
    def create_risk_analysis(self,
                           returns: pd.Series,
                           benchmark_returns: pd.Series,
                           save_path: str = 'outputs/visualizations/risk_analysis.png') -> None:
        """
        Create comprehensive risk analysis charts.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            save_path: Path to save the analysis
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. VaR Analysis
        var_95 = np.percentile(returns.dropna(), 5)
        var_99 = np.percentile(returns.dropna(), 1)
        
        axes[0, 0].hist(returns.dropna(), bins=50, alpha=0.7, color=self.colors['primary'], density=True)
        axes[0, 0].axvline(var_95, color=self.colors['danger'], linestyle='--', 
                          label=f'95% VaR: {var_95:.3f}')
        axes[0, 0].axvline(var_99, color=self.colors['loss'], linestyle='--', 
                          label=f'99% VaR: {var_99:.3f}')
        axes[0, 0].set_title('💰 Value at Risk Analysis', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Rolling Volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.rolling(30).std() * np.sqrt(252)
        
        axes[0, 1].plot(rolling_vol.index, rolling_vol.values, 
                       color=self.colors['primary'], label='Strategy Vol')
        axes[0, 1].plot(benchmark_vol.index, benchmark_vol.values,
                       color=self.colors['danger'], label='Benchmark Vol')
        axes[0, 1].set_title('📊 Rolling Volatility (30d)', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Risk-Return Scatter
        strategy_return = returns.mean() * 252
        strategy_vol = returns.std() * np.sqrt(252)
        benchmark_return = benchmark_returns.mean() * 252
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        axes[0, 2].scatter(strategy_vol, strategy_return, color=self.colors['profit'], 
                          s=100, label='Strategy', marker='o')
        axes[0, 2].scatter(benchmark_vol, benchmark_return, color=self.colors['danger'],
                          s=100, label='Benchmark', marker='s')
        axes[0, 2].set_xlabel('Volatility (Annualized)')
        axes[0, 2].set_ylabel('Return (Annualized)')
        axes[0, 2].set_title('📈 Risk-Return Profile', fontsize=12, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Beta Analysis
        if len(returns) > 30:
            beta = self._calculate_beta(returns, benchmark_returns)
            rolling_beta = self._calculate_rolling_beta(returns, benchmark_returns, window=30)
            
            axes[1, 0].plot(rolling_beta.index, rolling_beta.values, color=self.colors['secondary'])
            axes[1, 0].axhline(y=1, color=self.colors['neutral'], linestyle='--', alpha=0.7)
            axes[1, 0].axhline(y=beta, color=self.colors['danger'], linestyle='-', 
                              label=f'Overall Beta: {beta:.2f}')
            axes[1, 0].set_title(f'📊 Rolling Beta (30d)', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Tail Risk Analysis
        self._plot_tail_risk(axes[1, 1], returns)
        
        # 6. Risk Metrics Summary
        self._create_risk_metrics_table(axes[1, 2], returns, benchmark_returns)
        
        plt.suptitle('⚠️ Risk Analysis Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"⚠️ Risk analysis saved as '{save_path}'")
    
    # Helper methods
    def _plot_model_metrics(self, ax, metrics: Dict[str, float]) -> None:
        """Plot model performance metrics."""
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=[self.colors['primary'], 
                     self.colors['secondary'], self.colors['success']])
        ax.set_title('🤖 Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_importance(self, ax, feature_importance: pd.DataFrame) -> None:
        """Plot top feature importance."""
        top_10 = feature_importance.head(10)
        bars = ax.barh(range(len(top_10)), top_10['importance'])
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['feature'])
        ax.set_title('🔍 Top 10 Feature Importance', fontsize=14, fontweight='bold')
        
        # Color gradient
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / len(bars)))
        ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_returns(self, ax, backtest_results: Dict) -> None:
        """Plot cumulative returns comparison."""
        if 'strategy_returns' in backtest_results and 'benchmark_returns' in backtest_results:
            strategy_cum = (1 + backtest_results['strategy_returns']).cumprod()
            benchmark_cum = (1 + backtest_results['benchmark_returns']).cumprod()
            
            ax.plot(strategy_cum.index, strategy_cum.values, 
                   color=self.colors['profit'], linewidth=2, label='Strategy')
            ax.plot(benchmark_cum.index, benchmark_cum.values,
                   color=self.colors['danger'], linewidth=2, label='Benchmark')
            ax.set_title('📈 Cumulative Returns', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_risk_metrics(self, ax, backtest_results: Dict) -> None:
        """Plot risk metrics comparison."""
        metrics = ['sharpe_ratio', 'max_drawdown', 'volatility']
        strategy_metrics = [backtest_results.get(f'strategy_{m}', 0) for m in metrics]
        benchmark_metrics = [backtest_results.get(f'benchmark_{m}', 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, strategy_metrics, width, label='Strategy', color=self.colors['primary'])
        ax.bar(x + width/2, benchmark_metrics, width, label='Benchmark', color=self.colors['danger'])
        
        ax.set_xlabel('Risk Metrics')
        ax.set_title('⚠️ Risk Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_confidence(self, ax, probabilities: np.ndarray) -> None:
        """Plot prediction confidence distribution."""
        ax.hist(probabilities, bins=20, alpha=0.7, color=self.colors['secondary'], edgecolor='black')
        ax.axvline(0.5, color=self.colors['neutral'], linestyle='--', alpha=0.7, label='Neutral')
        ax.axvline(0.6, color=self.colors['success'], linestyle='--', alpha=0.7, label='Threshold')
        ax.set_title('🎯 Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrix(self, ax, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('📊 Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Down', 'Up'])
        ax.set_yticklabels(['Down', 'Up'])
    
    def _create_performance_table(self, ax, model_metrics: Dict, backtest_results: Dict) -> None:
        """Create performance summary table."""
        ax.axis('off')
        
        # Combine metrics
        all_metrics = {**model_metrics, **backtest_results}
        
        # Create table data
        table_data = []
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                if 'accuracy' in key.lower() or 'return' in key.lower():
                    formatted_value = f"{value:.1%}"
                elif 'ratio' in key.lower():
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.3f}"
                table_data.append([key.replace('_', ' ').title(), formatted_value])
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor(self.colors['primary'])
                    cell.set_text_props(color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('📋 Performance Summary', fontsize=14, fontweight='bold')
    
    def _plot_feature_categories(self, ax, feature_importance: pd.DataFrame) -> None:
        """Plot feature importance by categories."""
        # Categorize features
        categories = {
            'Technical': ['SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'ATR', 'Stoch', 'ADX', 'VWAP'],
            'Price': ['close', 'open', 'high', 'low', 'price', 'body', 'shadow'],
            'Volume': ['volume', 'OBV'],
            'Volatility': ['vol', 'atr'],
            'Time': ['hour', 'day', 'month', 'session'],
            'Statistical': ['mean', 'std', 'min', 'max', 'pct', 'lag', 'momentum']
        }
        
        feature_cats = {'Other': 0}
        for _, row in feature_importance.iterrows():
            feature_name = row['feature']
            categorized = False
            
            for cat, keywords in categories.items():
                if any(keyword.lower() in feature_name.lower() for keyword in keywords):
                    feature_cats[cat] = feature_cats.get(cat, 0) + row['importance']
                    categorized = True
                    break
            
            if not categorized:
                feature_cats['Other'] += row['importance']
        
        # Create pie chart
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], 
                 self.colors['info'], self.colors['neutral'], '#FF9800', '#9C27B0']
        
        wedges, texts, autotexts = ax.pie(feature_cats.values(), labels=feature_cats.keys(), 
                                         autopct='%1.1f%%', colors=colors[:len(feature_cats)])
        ax.set_title('📊 Feature Importance by Category', fontsize=14, fontweight='bold')
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        running_max = cumulative_returns.expanding().max()
        return (cumulative_returns - running_max) / running_max
    
    def _create_monthly_returns_matrix(self, monthly_returns: pd.Series) -> np.ndarray:
        """Create monthly returns matrix for heatmap."""
        # This is a simplified version - you might want to implement a more sophisticated one
        years = monthly_returns.index.year.unique()
        months = range(1, 13)
        
        matrix = np.full((len(years), 12), np.nan)
        
        for i, year in enumerate(years):
            year_data = monthly_returns[monthly_returns.index.year == year]
            for j, month in enumerate(months):
                month_data = year_data[year_data.index.month == month]
                if len(month_data) > 0:
                    matrix[i, j] = month_data.iloc[0]
        
        return matrix
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0
        
        covariance = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])[0, 1]
        benchmark_variance = np.var(aligned_data.iloc[:, 1])
        
        return covariance / benchmark_variance if benchmark_variance != 0 else 1.0
    
    def _calculate_rolling_beta(self, returns: pd.Series, benchmark_returns: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling beta."""
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        rolling_beta = aligned_data.rolling(window).cov().iloc[:, 0, 1] / aligned_data.iloc[:, 1].rolling(window).var()
        return rolling_beta.dropna()
    
    def _plot_tail_risk(self, ax, returns: pd.Series) -> None:
        """Plot tail risk analysis."""
        sorted_returns = returns.dropna().sort_values()
        tail_5_pct = sorted_returns[:int(len(sorted_returns) * 0.05)]
        
        ax.hist(tail_5_pct, bins=20, alpha=0.7, color=self.colors['danger'], 
               density=True, label='Worst 5% Returns')
        ax.axvline(tail_5_pct.mean(), color=self.colors['loss'], linestyle='--', 
                  label=f'Mean Tail Return: {tail_5_pct.mean():.3f}')
        ax.set_title('💀 Tail Risk Analysis', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_risk_metrics_table(self, ax, returns: pd.Series, benchmark_returns: pd.Series) -> None:
        """Create risk metrics summary table."""
        ax.axis('off')
        
        # Calculate risk metrics
        metrics_data = [
            ['Volatility', f"{returns.std() * np.sqrt(252):.1%}"],
            ['Sharpe Ratio', f"{returns.mean() / returns.std() * np.sqrt(252):.2f}"],
            ['Max Drawdown', f"{self._calculate_drawdown((1 + returns).cumprod()).min():.1%}"],
            ['VaR (95%)', f"{np.percentile(returns.dropna(), 5):.3f}"],
            ['Skewness', f"{returns.skew():.2f}"],
            ['Kurtosis', f"{returns.kurtosis():.2f}"]
        ]
        
        table = ax.table(cellText=metrics_data, colLabels=['Risk Metric', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax.set_title('📊 Risk Metrics Summary', fontsize=12, fontweight='bold')