"""
Advanced diagnostic visualization module for XGBoost cryptocurrency trading system.
Shows detailed model thinking process, learning evolution, and improvement insights.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import export_text
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Set style for professional diagnostic charts
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")


class XGBoostDiagnosticVisualizer:
    """
    Advanced diagnostic visualization suite for XGBoost model analysis.
    Reveals model thinking process and provides improvement insights.
    """

    def __init__(self, figsize: Tuple[int, int] = (20, 16), dpi: int = 300):
        """
        Initialize the diagnostic visualizer.

        Args:
            figsize: Default figure size for matplotlib charts
            dpi: Resolution for saved charts
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff9900",
            "info": "#17becf",
            "purple": "#9467bd",
            "brown": "#8c564b",
            "pink": "#e377c2",
            "gray": "#7f7f7f",
            "olive": "#bcbd22",
        }

    def create_comprehensive_diagnostic_dashboard(
        self,
        model: xgb.XGBClassifier,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        training_history: Optional[Dict] = None,
        save_path: str = "outputs/visualizations/xgboost_diagnostic_dashboard.png",
    ) -> None:
        """
        Create comprehensive diagnostic dashboard showing XGBoost thinking process.

        Args:
            model: Trained XGBoost model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            feature_names: List of feature names
            training_history: Optional training history data
            save_path: Path to save the dashboard
        """
        fig = plt.figure(figsize=(24, 20))

        # Create complex grid layout for comprehensive analysis
        gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)

        print("   🔍 Generating model learning curves...")
        # 1. Learning Curves (Top Row - Left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_learning_curves(ax1, model, X_train, y_train)

        print("   📊 Analyzing validation curves...")
        # 2. Validation Curves (Top Row - Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_validation_curves(ax2, X_train, y_train)

        print("   🎯 Creating prediction confidence analysis...")
        # 3. Prediction Confidence Analysis (Second Row - Left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_prediction_confidence_analysis(ax3, model, X_test, y_test)

        print("   🔬 Analyzing feature importance evolution...")
        # 4. Feature Importance Evolution (Second Row - Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_feature_importance_evolution(ax4, model, feature_names)

        print("   ❌ Creating error analysis...")
        # 5. Error Analysis (Third Row - Left)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_error_analysis(ax5, model, X_test, y_test, feature_names)

        print("   🌡️ Analyzing model calibration...")
        # 6. Model Calibration (Third Row - Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_model_calibration(ax6, model, X_test, y_test)

        print("   📈 Creating market regime analysis...")
        # 7. Market Regime Analysis (Fourth Row - Left)
        ax7 = fig.add_subplot(gs[3, :2])
        self._plot_market_regime_performance(ax7, model, X_test, y_test)

        print("   🔗 Analyzing feature interactions...")
        # 8. Feature Interactions (Fourth Row - Right)
        ax8 = fig.add_subplot(gs[3, 2:])
        self._plot_feature_interactions(ax8, model, X_train, feature_names)

        print("   🎛️ Creating hyperparameter sensitivity...")
        # 9. Hyperparameter Sensitivity (Bottom Row)
        ax9 = fig.add_subplot(gs[4, :])
        self._plot_hyperparameter_sensitivity(ax9, X_train, y_train)

        plt.suptitle(
            "🔬 XGBoost Comprehensive Diagnostic Dashboard\n"
            + "Deep Model Analysis & System Improvement Insights",
            fontsize=22,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print(f"🔬 Comprehensive diagnostic dashboard saved as '{save_path}'")

    def create_decision_boundary_analysis(
        self,
        model: xgb.XGBClassifier,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        save_path: str = "outputs/visualizations/decision_boundary_analysis.png",
    ) -> None:
        """
        Create detailed decision boundary and model behavior analysis.

        Args:
            model: Trained XGBoost model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            feature_names: List of feature names
            save_path: Path to save the analysis
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))

        print("   🎯 Analyzing decision boundaries...")

        # Get predictions and probabilities
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]

        # 1. 2D Decision Boundary (Top Left)
        self._plot_2d_decision_boundary(
            axes[0, 0], model, X_train, y_train, feature_names
        )

        # 2. Prediction Confidence Distribution (Top Middle)
        self._plot_confidence_distribution(
            axes[0, 1], y_proba_train, y_proba_test, y_train, y_test
        )

        # 3. Prediction vs Truth Scatter (Top Right)
        self._plot_prediction_truth_scatter(axes[0, 2], y_proba_test, y_test)

        # 4. Tree Depth Impact (Middle Left)
        self._plot_tree_depth_impact(axes[1, 0], model, X_test, y_test)

        # 5. Prediction Uncertainty (Middle Middle)
        self._plot_prediction_uncertainty(axes[1, 1], model, X_test, y_test)

        # 6. Feature Contribution Heatmap (Middle Right)
        self._plot_feature_contribution_heatmap(
            axes[1, 2], model, X_test, feature_names
        )

        # 7. Temporal Decision Patterns (Bottom Left)
        self._plot_temporal_decision_patterns(axes[2, 0], y_proba_test, y_test)

        # 8. Classification Boundary Stability (Bottom Middle)
        self._plot_boundary_stability(axes[2, 1], model, X_test)

        # 9. Model Confidence Evolution (Bottom Right)
        self._plot_confidence_evolution(axes[2, 2], y_proba_test)

        plt.suptitle(
            "🎯 XGBoost Decision Boundary & Behavior Analysis\n"
            + "Understanding Model Thinking Process",
            fontsize=18,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print(f"🎯 Decision boundary analysis saved as '{save_path}'")

    def create_feature_deep_dive(
        self,
        model: xgb.XGBClassifier,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        save_path: str = "outputs/visualizations/feature_deep_dive.png",
    ) -> None:
        """
        Create comprehensive feature analysis and selection insights.

        Args:
            model: Trained XGBoost model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            feature_names: List of feature names
            save_path: Path to save the analysis
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        print("   🔍 Creating comprehensive feature analysis...")

        # 1. Feature Importance by Type (Top Row)
        self._plot_feature_importance_by_type(axes[0, 0], model, feature_names)
        self._plot_feature_gain_vs_cover(axes[0, 1], model, feature_names)
        self._plot_feature_redundancy_analysis(axes[0, 2], X_train, feature_names)
        self._plot_feature_stability(axes[0, 3], model, X_train, y_train, feature_names)

        # 2. Feature Selection Impact (Middle Row)
        self._plot_feature_selection_impact(axes[1, 0], X_train, y_train, feature_names)
        self._plot_feature_correlation_network(axes[1, 1], X_train, feature_names)
        self._plot_feature_distribution_by_target(
            axes[1, 2], X_train, y_train, feature_names
        )
        self._plot_feature_outlier_impact(axes[1, 3], X_train, y_train, feature_names)

        # 3. Advanced Feature Insights (Bottom Row)
        self._plot_feature_interaction_strength(
            axes[2, 0], model, X_train, feature_names
        )
        self._plot_feature_prediction_contribution(
            axes[2, 1], model, X_test, feature_names
        )
        self._plot_feature_temporal_importance(axes[2, 2], X_train, feature_names)
        self._plot_feature_noise_analysis(axes[2, 3], X_train, y_train, feature_names)

        plt.suptitle(
            "🔍 Feature Engineering Deep Dive Analysis\n"
            + "Comprehensive Feature Selection & Optimization Insights",
            fontsize=18,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print(f"🔍 Feature deep dive saved as '{save_path}'")

    def create_performance_optimization_insights(
        self,
        model: xgb.XGBClassifier,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        save_path: str = "outputs/visualizations/optimization_insights.png",
    ) -> None:
        """
        Create performance optimization insights and recommendations.

        Args:
            model: Trained XGBoost model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            save_path: Path to save the insights
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        print("   ⚡ Generating optimization insights...")

        # 1. Overfitting Analysis (Top Row)
        self._plot_overfitting_analysis(
            axes[0, 0], model, X_train, X_test, y_train, y_test
        )
        self._plot_regularization_impact(axes[0, 1], X_train, y_train)
        self._plot_early_stopping_analysis(axes[0, 2], X_train, y_train)

        # 2. Performance Bottlenecks (Bottom Row)
        self._plot_training_efficiency(axes[1, 0], model)
        self._plot_prediction_speed_analysis(axes[1, 1], model, X_test)
        self._plot_improvement_recommendations(axes[1, 2], model, X_train, y_train)

        plt.suptitle(
            "⚡ XGBoost Performance Optimization Insights\n"
            + "System Improvement Recommendations",
            fontsize=18,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print(f"⚡ Optimization insights saved as '{save_path}'")

    # Helper methods for diagnostic visualizations
    def _plot_learning_curves(self, ax, model, X, y):
        """Plot learning curves showing training progression."""
        try:
            # Calculate learning curves
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model,
                X,
                y,
                train_sizes=train_sizes,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
            )

            # Calculate means and stds
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # Plot learning curves
            ax.plot(
                train_sizes_abs,
                train_mean,
                "o-",
                color=self.colors["primary"],
                label="Training Score",
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(
                train_sizes_abs,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
                color=self.colors["primary"],
            )

            ax.plot(
                train_sizes_abs,
                val_mean,
                "o-",
                color=self.colors["danger"],
                label="Validation Score",
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(
                train_sizes_abs,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.2,
                color=self.colors["danger"],
            )

            ax.set_xlabel("Training Set Size")
            ax.set_ylabel("Accuracy Score")
            ax.set_title("📈 Learning Curves Analysis", fontweight="bold", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add improvement suggestions
            gap = train_mean[-1] - val_mean[-1]
            if gap > 0.1:
                ax.text(
                    0.05,
                    0.95,
                    f"⚠️ Overfitting detected\nGap: {gap:.3f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Learning curves unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("📈 Learning Curves Analysis", fontweight="bold", fontsize=12)

    def _plot_validation_curves(self, ax, X, y):
        """Plot validation curves for hyperparameter analysis."""
        try:
            # Test different max_depth values
            param_range = [3, 4, 5, 6, 7, 8, 9, 10]
            train_scores, val_scores = validation_curve(
                xgb.XGBClassifier(random_state=42, verbosity=0),
                X,
                y,
                param_name="max_depth",
                param_range=param_range,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            ax.plot(
                param_range,
                train_mean,
                "o-",
                color=self.colors["primary"],
                label="Training Score",
                linewidth=2,
            )
            ax.fill_between(
                param_range,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
                color=self.colors["primary"],
            )

            ax.plot(
                param_range,
                val_mean,
                "o-",
                color=self.colors["danger"],
                label="Validation Score",
                linewidth=2,
            )
            ax.fill_between(
                param_range,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.2,
                color=self.colors["danger"],
            )

            # Find optimal depth
            optimal_depth = param_range[np.argmax(val_mean)]
            ax.axvline(
                optimal_depth,
                color=self.colors["success"],
                linestyle="--",
                label=f"Optimal Depth: {optimal_depth}",
            )

            ax.set_xlabel("Max Depth")
            ax.set_ylabel("Accuracy Score")
            ax.set_title(
                "🎛️ Hyperparameter Validation Curves", fontweight="bold", fontsize=12
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Validation curves unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                "🎛️ Hyperparameter Validation Curves", fontweight="bold", fontsize=12
            )

    def _plot_prediction_confidence_analysis(self, ax, model, X_test, y_test):
        """Plot prediction confidence analysis."""
        try:
            y_proba = model.predict_proba(X_test)[:, 1]

            # Create confidence bins
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_accuracies = []
            bin_counts = []

            for i in range(len(bins) - 1):
                mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
                if mask.sum() > 0:
                    bin_accuracy = y_test[mask].mean()
                    bin_accuracies.append(bin_accuracy)
                    bin_counts.append(mask.sum())
                else:
                    bin_accuracies.append(0)
                    bin_counts.append(0)

            # Plot calibration
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
            ax.plot(
                bin_centers,
                bin_accuracies,
                "o-",
                color=self.colors["primary"],
                linewidth=2,
                markersize=8,
                label="Model Calibration",
            )

            # Add size information
            for i, (x, y, count) in enumerate(
                zip(bin_centers, bin_accuracies, bin_counts)
            ):
                if count > 0:
                    ax.annotate(
                        f"{count}",
                        (x, y),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(
                "🎯 Prediction Confidence Calibration", fontweight="bold", fontsize=12
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Calculate calibration error
            cal_error = np.mean(np.abs(np.array(bin_accuracies) - bin_centers))
            ax.text(
                0.05,
                0.95,
                f"Calibration Error: {cal_error:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Confidence analysis unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                "🎯 Prediction Confidence Analysis", fontweight="bold", fontsize=12
            )

    def _plot_feature_importance_evolution(self, ax, model, feature_names):
        """Plot feature importance with detailed breakdown."""
        try:
            # Get different importance types
            importance_gain = model.get_booster().get_score(importance_type="gain")
            importance_cover = model.get_booster().get_score(importance_type="cover")
            importance_freq = model.get_booster().get_score(importance_type="frequency")

            # Get top 10 features by gain
            top_features = sorted(
                importance_gain.items(), key=lambda x: x[1], reverse=True
            )[:10]

            if top_features:
                features = [f[0] for f in top_features]
                gains = [importance_gain.get(f, 0) for f in features]
                covers = [importance_cover.get(f, 0) for f in features]
                freqs = [importance_freq.get(f, 0) for f in features]

                # Normalize values
                gains = (
                    np.array(gains) / max(gains) if max(gains) > 0 else np.array(gains)
                )
                covers = (
                    np.array(covers) / max(covers)
                    if max(covers) > 0
                    else np.array(covers)
                )
                freqs = (
                    np.array(freqs) / max(freqs) if max(freqs) > 0 else np.array(freqs)
                )

                x = np.arange(len(features))
                width = 0.25

                ax.bar(
                    x - width,
                    gains,
                    width,
                    label="Gain",
                    color=self.colors["primary"],
                    alpha=0.8,
                )
                ax.bar(
                    x,
                    covers,
                    width,
                    label="Cover",
                    color=self.colors["secondary"],
                    alpha=0.8,
                )
                ax.bar(
                    x + width,
                    freqs,
                    width,
                    label="Frequency",
                    color=self.colors["success"],
                    alpha=0.8,
                )

                ax.set_xlabel("Features")
                ax.set_ylabel("Normalized Importance")
                ax.set_title(
                    "🔬 Feature Importance Evolution", fontweight="bold", fontsize=12
                )
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [f[:10] + "..." if len(f) > 10 else f for f in features],
                    rotation=45,
                )
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No feature importance data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(
                    "🔬 Feature Importance Evolution", fontweight="bold", fontsize=12
                )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Feature importance unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                "🔬 Feature Importance Evolution", fontweight="bold", fontsize=12
            )

    def _plot_error_analysis(self, ax, model, X_test, y_test, feature_names):
        """Plot detailed error analysis."""
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Identify different types of errors
            true_positives = (y_test == 1) & (y_pred == 1)
            false_positives = (y_test == 0) & (y_pred == 1)
            true_negatives = (y_test == 0) & (y_pred == 0)
            false_negatives = (y_test == 1) & (y_pred == 0)

            # Plot confidence distribution for each category
            categories = [
                "True Positive",
                "False Positive",
                "True Negative",
                "False Negative",
            ]
            masks = [true_positives, false_positives, true_negatives, false_negatives]
            colors = [
                self.colors["success"],
                self.colors["warning"],
                self.colors["info"],
                self.colors["danger"],
            ]

            for i, (category, mask, color) in enumerate(zip(categories, masks, colors)):
                if mask.sum() > 0:
                    confidence = y_proba[mask]
                    ax.hist(
                        confidence,
                        bins=20,
                        alpha=0.6,
                        label=f"{category} ({mask.sum()})",
                        color=color,
                        density=True,
                    )

            ax.set_xlabel("Prediction Confidence")
            ax.set_ylabel("Density")
            ax.set_title(
                "❌ Error Analysis by Confidence", fontweight="bold", fontsize=12
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add error rates
            fp_rate = (
                false_positives.sum() / (false_positives.sum() + true_negatives.sum())
                if (false_positives.sum() + true_negatives.sum()) > 0
                else 0
            )
            fn_rate = (
                false_negatives.sum() / (false_negatives.sum() + true_positives.sum())
                if (false_negatives.sum() + true_positives.sum()) > 0
                else 0
            )

            ax.text(
                0.05,
                0.95,
                f"FP Rate: {fp_rate:.3f}\nFN Rate: {fn_rate:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
            )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error analysis unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("❌ Error Analysis", fontweight="bold", fontsize=12)

    def _plot_model_calibration(self, ax, model, X_test, y_test):
        """Plot model calibration analysis."""
        try:
            from sklearn.calibration import calibration_curve

            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_proba, n_bins=10, normalize=False
            )

            # Plot calibration curve
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                "o-",
                color=self.colors["primary"],
                linewidth=2,
                markersize=8,
                label="Model",
            )

            # Calculate Brier score
            brier_score = np.mean((y_proba - y_test) ** 2)

            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("🌡️ Model Calibration Analysis", fontweight="bold", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax.text(
                0.05,
                0.95,
                f"Brier Score: {brier_score:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Calibration analysis unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("🌡️ Model Calibration Analysis", fontweight="bold", fontsize=12)

    def _plot_market_regime_performance(self, ax, model, X_test, y_test):
        """Plot performance across different market regimes."""
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Create artificial market regimes based on volatility or other features
            # This is a simplified example - you might want to use actual market regime indicators
            if len(X_test) > 20:
                volatility_feature = None
                for col in X_test.columns:
                    if "vol" in col.lower() or "atr" in col.lower():
                        volatility_feature = col
                        break

                if volatility_feature:
                    vol_values = X_test[volatility_feature]
                    vol_terciles = np.percentile(vol_values, [33, 67])

                    low_vol = vol_values <= vol_terciles[0]
                    med_vol = (vol_values > vol_terciles[0]) & (
                        vol_values <= vol_terciles[1]
                    )
                    high_vol = vol_values > vol_terciles[1]

                    regimes = ["Low Vol", "Medium Vol", "High Vol"]
                    regime_masks = [low_vol, med_vol, high_vol]

                    accuracies = []
                    sample_counts = []

                    for mask in regime_masks:
                        if mask.sum() > 0:
                            accuracy = (y_test[mask] == y_pred[mask]).mean()
                            accuracies.append(accuracy)
                            sample_counts.append(mask.sum())
                        else:
                            accuracies.append(0)
                            sample_counts.append(0)

                    bars = ax.bar(
                        regimes,
                        accuracies,
                        color=[
                            self.colors["success"],
                            self.colors["warning"],
                            self.colors["danger"],
                        ],
                        alpha=0.7,
                    )

                    # Add sample counts on bars
                    for bar, count in zip(bars, sample_counts):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.01,
                            f"n={count}",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                        )

                    ax.set_ylabel("Accuracy")
                    ax.set_title(
                        "📈 Performance Across Market Regimes",
                        fontweight="bold",
                        fontsize=12,
                    )
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No volatility features found\nfor regime analysis",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(
                        "📈 Market Regime Performance", fontweight="bold", fontsize=12
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data\nfor regime analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(
                    "📈 Market Regime Performance", fontweight="bold", fontsize=12
                )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Market regime analysis unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("📈 Market Regime Performance", fontweight="bold", fontsize=12)

    def _plot_feature_interactions(self, ax, model, X_train, feature_names):
        """Plot feature interaction analysis."""
        try:
            # Get feature importance
            importance = model.feature_importances_
            top_indices = np.argsort(importance)[-10:]  # Top 10 features

            if len(top_indices) > 1:
                top_features = [feature_names[i] for i in top_indices]
                interaction_matrix = np.zeros((len(top_features), len(top_features)))

                # Calculate simple correlation as proxy for interaction
                top_feature_data = X_train.iloc[:, top_indices]
                correlation_matrix = top_feature_data.corr().values

                # Plot heatmap
                im = ax.imshow(np.abs(correlation_matrix), cmap="Blues", aspect="auto")

                # Add labels
                ax.set_xticks(range(len(top_features)))
                ax.set_yticks(range(len(top_features)))
                ax.set_xticklabels(
                    [f[:8] + "..." if len(f) > 8 else f for f in top_features],
                    rotation=45,
                )
                ax.set_yticklabels(
                    [f[:8] + "..." if len(f) > 8 else f for f in top_features]
                )

                # Add correlation values
                for i in range(len(top_features)):
                    for j in range(len(top_features)):
                        text = ax.text(
                            j,
                            i,
                            f"{correlation_matrix[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=8,
                        )

                ax.set_title(
                    "🔗 Feature Interactions (Correlation)",
                    fontweight="bold",
                    fontsize=12,
                )
                plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient features\nfor interaction analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("🔗 Feature Interactions", fontweight="bold", fontsize=12)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Feature interaction analysis unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("🔗 Feature Interactions", fontweight="bold", fontsize=12)

    def _plot_hyperparameter_sensitivity(self, ax, X_train, y_train):
        """Plot hyperparameter sensitivity analysis."""
        try:
            # Test different hyperparameter combinations
            learning_rates = [0.05, 0.1, 0.2, 0.3]
            max_depths = [4, 6, 8, 10]

            results = np.zeros((len(learning_rates), len(max_depths)))

            for i, lr in enumerate(learning_rates):
                for j, depth in enumerate(max_depths):
                    try:
                        # Quick cross-validation
                        model = xgb.XGBClassifier(
                            learning_rate=lr,
                            max_depth=depth,
                            n_estimators=50,
                            random_state=42,
                            verbosity=0,
                        )
                        from sklearn.model_selection import cross_val_score

                        scores = cross_val_score(
                            model, X_train, y_train, cv=3, scoring="accuracy"
                        )
                        results[i, j] = scores.mean()
                    except:
                        results[i, j] = 0

            # Plot heatmap
            im = ax.imshow(results, cmap="RdYlGn", aspect="auto")

            # Add labels
            ax.set_xticks(range(len(max_depths)))
            ax.set_yticks(range(len(learning_rates)))
            ax.set_xticklabels([f"Depth: {d}" for d in max_depths])
            ax.set_yticklabels([f"LR: {lr}" for lr in learning_rates])

            # Add values
            for i in range(len(learning_rates)):
                for j in range(len(max_depths)):
                    text = ax.text(
                        j,
                        i,
                        f"{results[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=10,
                    )

            ax.set_title(
                "🎛️ Hyperparameter Sensitivity Analysis", fontweight="bold", fontsize=12
            )
            ax.set_xlabel("Max Depth")
            ax.set_ylabel("Learning Rate")
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Find and highlight best combination
            best_i, best_j = np.unravel_index(np.argmax(results), results.shape)
            rect = patches.Rectangle(
                (best_j - 0.5, best_i - 0.5),
                1,
                1,
                linewidth=3,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Hyperparameter analysis unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("🎛️ Hyperparameter Sensitivity", fontweight="bold", fontsize=12)

    # Additional helper methods for decision boundary analysis
    def _plot_2d_decision_boundary(self, ax, model, X_train, y_train, feature_names):
        """Plot 2D decision boundary using top 2 features."""
        try:
            # Get top 2 most important features
            importance = model.feature_importances_
            top_2_indices = np.argsort(importance)[-2:]

            if len(top_2_indices) >= 2:
                X_2d = X_train.iloc[:, top_2_indices]
                feature_1, feature_2 = (
                    feature_names[top_2_indices[0]],
                    feature_names[top_2_indices[1]],
                )

                # Create a mesh for decision boundary
                h = (X_2d.iloc[:, 0].max() - X_2d.iloc[:, 0].min()) / 100
                xx, yy = np.meshgrid(
                    np.arange(X_2d.iloc[:, 0].min(), X_2d.iloc[:, 0].max(), h),
                    np.arange(X_2d.iloc[:, 1].min(), X_2d.iloc[:, 1].max(), h),
                )

                # Create a temporary model for 2D visualization
                model_2d = xgb.XGBClassifier(random_state=42, verbosity=0)
                model_2d.fit(X_2d, y_train)

                # Predict on mesh
                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                Z = model_2d.predict_proba(mesh_points)[:, 1]
                Z = Z.reshape(xx.shape)

                # Plot decision boundary
                ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap="RdYlBu")

                # Plot data points
                scatter = ax.scatter(
                    X_2d.iloc[:, 0],
                    X_2d.iloc[:, 1],
                    c=y_train,
                    cmap="RdYlBu",
                    edgecolors="black",
                    alpha=0.8,
                )

                ax.set_xlabel(
                    feature_1[:15] + "..." if len(feature_1) > 15 else feature_1
                )
                ax.set_ylabel(
                    feature_2[:15] + "..." if len(feature_2) > 15 else feature_2
                )
                ax.set_title("🎯 2D Decision Boundary", fontweight="bold", fontsize=12)
                plt.colorbar(scatter, ax=ax, shrink=0.8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient features\nfor 2D boundary",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("🎯 2D Decision Boundary", fontweight="bold", fontsize=12)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Decision boundary unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("🎯 2D Decision Boundary", fontweight="bold", fontsize=12)

    def _plot_confidence_distribution(
        self, ax, y_proba_train, y_proba_test, y_train, y_test
    ):
        """Plot confidence distribution for train and test sets."""
        try:
            ax.hist(
                y_proba_train,
                bins=30,
                alpha=0.6,
                label=f"Train (n={len(y_proba_train)})",
                color=self.colors["primary"],
                density=True,
            )
            ax.hist(
                y_proba_test,
                bins=30,
                alpha=0.6,
                label=f"Test (n={len(y_proba_test)})",
                color=self.colors["danger"],
                density=True,
            )

            # Add vertical lines for different confidence levels
            for conf, label in [(0.3, "Low"), (0.5, "Medium"), (0.7, "High")]:
                ax.axvline(conf, linestyle="--", alpha=0.7, label=f"{label} Conf")

            ax.set_xlabel("Prediction Probability")
            ax.set_ylabel("Density")
            ax.set_title("📊 Confidence Distribution", fontweight="bold", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Confidence distribution unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("📊 Confidence Distribution", fontweight="bold", fontsize=12)

    def _plot_prediction_truth_scatter(self, ax, y_proba_test, y_test):
        """Plot scatter of predictions vs truth."""
        try:
            # Add some jitter for visualization
            y_test_jitter = y_test + np.random.normal(0, 0.05, len(y_test))

            ax.scatter(
                y_proba_test, y_test_jitter, alpha=0.6, color=self.colors["primary"]
            )
            ax.axhline(
                y=0.5, color="red", linestyle="--", alpha=0.7, label="Decision Boundary"
            )
            ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7)

            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("True Label (with jitter)")
            ax.set_title("🎯 Prediction vs Truth", fontweight="bold", fontsize=12)
            ax.set_ylim(-0.5, 1.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Prediction scatter unavailable\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("🎯 Prediction vs Truth", fontweight="bold", fontsize=12)

    # Placeholder methods for additional analyses (to be implemented)
    def _plot_tree_depth_impact(self, ax, model, X_test, y_test):
        """Plot impact of tree depth on predictions."""
        ax.text(
            0.5,
            0.5,
            "Tree Depth Impact\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🌳 Tree Depth Impact", fontweight="bold", fontsize=12)

    def _plot_prediction_uncertainty(self, ax, model, X_test, y_test):
        """Plot prediction uncertainty analysis."""
        ax.text(
            0.5,
            0.5,
            "Prediction Uncertainty\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🔮 Prediction Uncertainty", fontweight="bold", fontsize=12)

    def _plot_feature_contribution_heatmap(self, ax, model, X_test, feature_names):
        """Plot feature contribution heatmap."""
        ax.text(
            0.5,
            0.5,
            "Feature Contribution\nHeatmap",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🔥 Feature Contributions", fontweight="bold", fontsize=12)

    def _plot_temporal_decision_patterns(self, ax, y_proba_test, y_test):
        """Plot temporal patterns in decisions."""
        ax.text(
            0.5,
            0.5,
            "Temporal Decision\nPatterns",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("⏰ Temporal Patterns", fontweight="bold", fontsize=12)

    def _plot_boundary_stability(self, ax, model, X_test):
        """Plot decision boundary stability."""
        ax.text(
            0.5,
            0.5,
            "Boundary Stability\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🏔️ Boundary Stability", fontweight="bold", fontsize=12)

    def _plot_confidence_evolution(self, ax, y_proba_test):
        """Plot how confidence evolves over time."""
        ax.text(
            0.5,
            0.5,
            "Confidence Evolution\nOver Time",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("📈 Confidence Evolution", fontweight="bold", fontsize=12)

    # Implemented methods for feature deep dive analysis
    def _plot_feature_importance_by_type(self, ax, model, feature_names):
        """Plot feature importance by feature type."""
        try:
            importance = model.feature_importances_

            # Categorize features
            categories = {
                "Technical": [
                    "SMA",
                    "EMA",
                    "RSI",
                    "MACD",
                    "BB_",
                    "ATR",
                    "Stoch",
                    "ADX",
                    "VWAP",
                    "Williams",
                    "CCI",
                    "ROC",
                    "Momentum",
                    "OBV",
                ],
                "Price": [
                    "close",
                    "open",
                    "high",
                    "low",
                    "price",
                    "body",
                    "shadow",
                    "range",
                ],
                "Volume": ["volume", "OBV"],
                "Volatility": ["vol", "atr"],
                "Time": ["hour", "day", "month", "session", "weekend"],
                "Statistical": [
                    "mean",
                    "std",
                    "min",
                    "max",
                    "pct",
                    "lag",
                    "momentum",
                    "median",
                    "skew",
                    "kurt",
                ],
            }

            feature_type_importance = {"Other": 0}
            for i, feature_name in enumerate(feature_names):
                categorized = False
                for cat, keywords in categories.items():
                    if any(
                        keyword.lower() in feature_name.lower() for keyword in keywords
                    ):
                        feature_type_importance[cat] = (
                            feature_type_importance.get(cat, 0) + importance[i]
                        )
                        categorized = True
                        break
                if not categorized:
                    feature_type_importance["Other"] += importance[i]

            # Plot bar chart
            types = list(feature_type_importance.keys())
            importances = list(feature_type_importance.values())

            bars = ax.bar(
                types,
                importances,
                color=[
                    self.colors["primary"],
                    self.colors["secondary"],
                    self.colors["success"],
                    self.colors["danger"],
                    self.colors["warning"],
                    self.colors["info"],
                ][: len(types)],
            )

            ax.set_title("📊 Importance by Type", fontweight="bold", fontsize=10)
            ax.set_ylabel("Total Importance")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Feature type analysis\nunavailable: {str(e)[:30]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("📊 Importance by Type", fontweight="bold", fontsize=10)

    def _plot_feature_gain_vs_cover(self, ax, model, feature_names):
        """Plot feature gain vs cover analysis."""
        try:
            # Get different importance types from XGBoost
            importance_gain = model.get_booster().get_score(importance_type="gain")
            importance_cover = model.get_booster().get_score(importance_type="cover")

            # Extract values for features that exist in both
            common_features = set(importance_gain.keys()) & set(importance_cover.keys())

            if common_features:
                gains = [importance_gain[f] for f in common_features]
                covers = [importance_cover[f] for f in common_features]

                # Create scatter plot
                scatter = ax.scatter(
                    gains, covers, alpha=0.7, c=range(len(gains)), cmap="viridis", s=60
                )

                # Add feature labels for top features
                top_features = sorted(
                    zip(common_features, gains, covers),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]

                for feature, gain, cover in top_features:
                    ax.annotate(
                        feature[:8] + "..." if len(feature) > 8 else feature,
                        (gain, cover),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )

                ax.set_xlabel("Gain")
                ax.set_ylabel("Cover")
                ax.set_title("📈 Gain vs Cover", fontweight="bold", fontsize=10)
                ax.grid(True, alpha=0.3)

                # Add correlation info
                if len(gains) > 1:
                    corr = np.corrcoef(gains, covers)[0, 1]
                    ax.text(
                        0.05,
                        0.95,
                        f"Correlation: {corr:.3f}",
                        transform=ax.transAxes,
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                        ),
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No gain/cover data\navailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("📈 Gain vs Cover", fontweight="bold", fontsize=10)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Gain vs cover analysis\nunavailable: {str(e)[:30]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("📈 Gain vs Cover", fontweight="bold", fontsize=10)

    def _plot_feature_redundancy_analysis(self, ax, X_train, feature_names):
        """Plot feature redundancy analysis."""
        try:
            # Calculate correlation matrix
            corr_matrix = X_train.corr().abs()

            # Find highly correlated feature pairs (> 0.8)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr_pairs.append(
                            (
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j],
                            )
                        )

            if high_corr_pairs:
                # Plot correlation distribution
                all_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        all_correlations.append(corr_matrix.iloc[i, j])

                ax.hist(
                    all_correlations,
                    bins=20,
                    alpha=0.7,
                    color=self.colors["primary"],
                    edgecolor="black",
                )
                ax.axvline(
                    0.8,
                    color=self.colors["danger"],
                    linestyle="--",
                    label=f"High Correlation\n({len(high_corr_pairs)} pairs)",
                )

                ax.set_xlabel("Absolute Correlation")
                ax.set_ylabel("Frequency")
                ax.set_title("🔄 Redundancy Analysis", fontweight="bold", fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                # Add summary text
                redundancy_pct = (
                    len(high_corr_pairs)
                    / (len(feature_names) * (len(feature_names) - 1) / 2)
                    * 100
                )
                ax.text(
                    0.05,
                    0.95,
                    f"Redundancy: {redundancy_pct:.1f}%",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No high correlation\nfeatures found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("🔄 Redundancy Analysis", fontweight="bold", fontsize=10)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Redundancy analysis\nunavailable: {str(e)[:30]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("🔄 Redundancy Analysis", fontweight="bold", fontsize=10)

    def _plot_feature_stability(self, ax, model, X_train, y_train, feature_names):
        """Plot feature stability across different samples."""
        try:
            # Calculate feature stability using bootstrap sampling
            from sklearn.utils import resample
            import xgboost as xgb

            n_bootstrap = 5  # Reduced for speed
            feature_importance_variations = []

            for i in range(n_bootstrap):
                # Bootstrap sample
                X_boot, y_boot = resample(X_train, y_train, random_state=i)

                # Train simple model
                model_boot = xgb.XGBClassifier(
                    n_estimators=50, max_depth=4, random_state=i, verbosity=0
                )
                model_boot.fit(X_boot, y_boot)

                feature_importance_variations.append(model_boot.feature_importances_)

            # Calculate stability (coefficient of variation)
            if feature_importance_variations:
                importance_array = np.array(feature_importance_variations)
                stability_scores = 1 - (
                    np.std(importance_array, axis=0)
                    / (np.mean(importance_array, axis=0) + 1e-8)
                )

                # Plot top 10 features
                top_10_indices = np.argsort(np.mean(importance_array, axis=0))[-10:]
                top_10_stability = stability_scores[top_10_indices]
                top_10_names = [feature_names[i] for i in top_10_indices]

                bars = ax.barh(
                    range(len(top_10_stability)),
                    top_10_stability,
                    color=self.colors["success"],
                    alpha=0.7,
                )

                ax.set_yticks(range(len(top_10_names)))
                ax.set_yticklabels(
                    [f[:10] + ".." if len(f) > 10 else f for f in top_10_names],
                    fontsize=8,
                )
                ax.set_xlabel("Stability Score")
                ax.set_title("⚖️ Feature Stability", fontweight="bold", fontsize=10)
                ax.grid(True, alpha=0.3)

                # Add stability threshold
                ax.axvline(
                    x=0.8,
                    color=self.colors["danger"],
                    linestyle="--",
                    alpha=0.7,
                    label="Stability threshold",
                )
                ax.legend(fontsize=8)

                # Color bars based on stability
                for i, bar in enumerate(bars):
                    if top_10_stability[i] < 0.8:
                        bar.set_color(self.colors["warning"])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Stability analysis\nunavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("⚖️ Feature Stability", fontweight="bold", fontsize=10)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Stability analysis\nunavailable: {str(e)[:30]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("⚖️ Feature Stability", fontweight="bold", fontsize=10)

    def _plot_feature_selection_impact(self, ax, X_train, y_train, feature_names):
        """Plot impact of feature selection."""
        try:
            from sklearn.feature_selection import SelectKBest, f_classif

            # Test different numbers of features
            n_features_range = [5, 10, 15, 20, 25, 30, min(35, len(feature_names))]
            scores = []

            for n_features in n_features_range:
                if n_features <= len(feature_names):
                    try:
                        # Quick model evaluation with selected features
                        selector = SelectKBest(score_func=f_classif, k=n_features)
                        X_selected = selector.fit_transform(X_train, y_train)

                        # Simple cross-validation score
                        from sklearn.model_selection import cross_val_score
                        from sklearn.ensemble import RandomForestClassifier

                        model = RandomForestClassifier(n_estimators=50, random_state=42)
                        cv_scores = cross_val_score(
                            model, X_selected, y_train, cv=3, scoring="accuracy"
                        )
                        scores.append(cv_scores.mean())
                    except:
                        scores.append(0)
                else:
                    scores.append(0)

            # Plot the impact
            ax.plot(
                n_features_range,
                scores,
                "o-",
                color=self.colors["primary"],
                linewidth=2,
                markersize=6,
            )

            # Find optimal number of features
            if scores:
                optimal_idx = np.argmax(scores)
                optimal_n = n_features_range[optimal_idx]
                ax.axvline(
                    optimal_n,
                    color=self.colors["success"],
                    linestyle="--",
                    label=f"Optimal: {optimal_n} features",
                )

            ax.set_xlabel("Number of Features")
            ax.set_ylabel("CV Accuracy")
            ax.set_title("🎯 Selection Impact", fontweight="bold", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Selection impact\nunavailable: {str(e)[:30]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("🎯 Selection Impact", fontweight="bold", fontsize=10)

    def _plot_feature_correlation_network(self, ax, X_train, feature_names):
        """Plot feature correlation network."""
        try:
            # Calculate correlation matrix for top 10 features
            from sklearn.feature_selection import mutual_info_classif

            # Get top 10 features by correlation strength
            corr_matrix = X_train.corr().abs()
            mean_corr = corr_matrix.mean().sort_values(ascending=False)
            top_10_features = mean_corr.head(10).index

            if len(top_10_features) > 1:
                # Create correlation network
                subset_corr = corr_matrix.loc[top_10_features, top_10_features]

                # Plot as heatmap
                im = ax.imshow(
                    subset_corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1
                )

                # Set labels
                ax.set_xticks(range(len(top_10_features)))
                ax.set_yticks(range(len(top_10_features)))
                ax.set_xticklabels(
                    [f[:8] + ".." if len(f) > 8 else f for f in top_10_features],
                    rotation=45,
                    fontsize=8,
                )
                ax.set_yticklabels(
                    [f[:8] + ".." if len(f) > 8 else f for f in top_10_features],
                    fontsize=8,
                )

                # Add correlation values
                for i in range(len(top_10_features)):
                    for j in range(len(top_10_features)):
                        text = ax.text(
                            j,
                            i,
                            f"{subset_corr.iloc[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=6,
                        )

                ax.set_title("🕸️ Correlation Network", fontweight="bold", fontsize=10)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Correlation", rotation=270, labelpad=15, fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient features\nfor network analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("🕸️ Correlation Network", fontweight="bold", fontsize=10)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Correlation network\nunavailable: {str(e)[:30]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("🕸️ Correlation Network", fontweight="bold", fontsize=10)

    def _plot_feature_distribution_by_target(self, ax, X_train, y_train, feature_names):
        """Plot feature distribution by target class."""
        try:
            # Select top 3 features for visualization
            from sklearn.feature_selection import mutual_info_classif

            mi_scores = mutual_info_classif(X_train, y_train)
            top_3_indices = np.argsort(mi_scores)[-3:]

            if len(top_3_indices) > 0:
                # Create subplot for each top feature
                feature_idx = top_3_indices[-1]  # Most important feature
                feature_name = feature_names[feature_idx]
                feature_values = X_train.iloc[:, feature_idx]

                # Plot distributions for each class
                class_0_values = feature_values[y_train == 0]
                class_1_values = feature_values[y_train == 1]

                ax.hist(
                    class_0_values,
                    bins=20,
                    alpha=0.6,
                    label="Class 0 (Down)",
                    color=self.colors["danger"],
                    density=True,
                )
                ax.hist(
                    class_1_values,
                    bins=20,
                    alpha=0.6,
                    label="Class 1 (Up)",
                    color=self.colors["success"],
                    density=True,
                )

                ax.set_xlabel("Feature Value")
                ax.set_ylabel("Density")
                ax.set_title(
                    f"📊 {feature_name[:15]}...", fontweight="bold", fontsize=10
                )
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                # Add KS test statistic
                from scipy import stats

                ks_stat, p_value = stats.ks_2samp(class_0_values, class_1_values)
                ax.text(
                    0.05,
                    0.95,
                    f"KS: {ks_stat:.3f}\np: {p_value:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No features available\nfor distribution analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(
                    "📊 Distribution by Target", fontweight="bold", fontsize=10
                )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Distribution analysis\nunavailable: {str(e)[:30]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("📊 Distribution by Target", fontweight="bold", fontsize=10)

    def _plot_feature_outlier_impact(self, ax, X_train, y_train, feature_names):
        """Plot impact of feature outliers."""
        try:
            # Calculate outlier percentage for each feature
            outlier_percentages = []
            feature_subset = X_train.iloc[
                :, : min(10, len(feature_names))
            ]  # Top 10 features

            for col in feature_subset.columns:
                Q1 = feature_subset[col].quantile(0.25)
                Q3 = feature_subset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (feature_subset[col] < lower_bound) | (
                    feature_subset[col] > upper_bound
                )
                outlier_pct = outliers.mean() * 100
                outlier_percentages.append(outlier_pct)

            if outlier_percentages:
                # Create bar plot
                feature_subset_names = [
                    f[:8] + ".." if len(f) > 8 else f for f in feature_subset.columns
                ]
                bars = ax.bar(
                    range(len(outlier_percentages)),
                    outlier_percentages,
                    color=self.colors["warning"],
                    alpha=0.7,
                )

                ax.set_xlabel("Features")
                ax.set_ylabel("Outlier Percentage (%)")
                ax.set_title("⚡ Outlier Impact", fontweight="bold", fontsize=10)
                ax.set_xticks(range(len(feature_subset_names)))
                ax.set_xticklabels(feature_subset_names, rotation=45, fontsize=8)
                ax.grid(True, alpha=0.3)

                # Add percentage labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.5,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                # Add threshold line
                ax.axhline(
                    y=5,
                    color=self.colors["danger"],
                    linestyle="--",
                    alpha=0.7,
                    label="5% threshold",
                )
                ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No outlier data\navailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("⚡ Outlier Impact", fontweight="bold", fontsize=10)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Outlier analysis\nunavailable: {str(e)[:30]}...",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_title("⚡ Outlier Impact", fontweight="bold", fontsize=10)

    def _plot_feature_interaction_strength(self, ax, model, X_train, feature_names):
        """Plot feature interaction strength."""
        ax.text(
            0.5,
            0.5,
            "Feature Interaction\nStrength",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🔗 Interaction Strength", fontweight="bold", fontsize=10)

    def _plot_feature_prediction_contribution(self, ax, model, X_test, feature_names):
        """Plot individual feature contributions to predictions."""
        ax.text(
            0.5,
            0.5,
            "Feature Prediction\nContribution",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🎯 Prediction Contribution", fontweight="bold", fontsize=10)

    def _plot_feature_temporal_importance(self, ax, X_train, feature_names):
        """Plot how feature importance changes over time."""
        ax.text(
            0.5,
            0.5,
            "Feature Temporal\nImportance",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("⏰ Temporal Importance", fontweight="bold", fontsize=10)

    def _plot_feature_noise_analysis(self, ax, X_train, y_train, feature_names):
        """Plot feature noise analysis."""
        ax.text(
            0.5,
            0.5,
            "Feature Noise\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🔊 Noise Analysis", fontweight="bold", fontsize=10)

    def _plot_overfitting_analysis(self, ax, model, X_train, X_test, y_train, y_test):
        """Plot overfitting analysis."""
        ax.text(
            0.5,
            0.5,
            "Overfitting\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("📈 Overfitting Analysis", fontweight="bold", fontsize=12)

    def _plot_regularization_impact(self, ax, X_train, y_train):
        """Plot regularization impact."""
        ax.text(
            0.5,
            0.5,
            "Regularization\nImpact",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🎛️ Regularization Impact", fontweight="bold", fontsize=12)

    def _plot_early_stopping_analysis(self, ax, X_train, y_train):
        """Plot early stopping analysis."""
        ax.text(
            0.5,
            0.5,
            "Early Stopping\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("⏹️ Early Stopping", fontweight="bold", fontsize=12)

    def _plot_training_efficiency(self, ax, model):
        """Plot training efficiency metrics."""
        ax.text(
            0.5,
            0.5,
            "Training\nEfficiency",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("⚡ Training Efficiency", fontweight="bold", fontsize=12)

    def _plot_prediction_speed_analysis(self, ax, model, X_test):
        """Plot prediction speed analysis."""
        ax.text(
            0.5,
            0.5,
            "Prediction Speed\nAnalysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("🏃 Prediction Speed", fontweight="bold", fontsize=12)

    def _plot_improvement_recommendations(self, ax, model, X_train, y_train):
        """Plot improvement recommendations."""
        ax.text(
            0.5,
            0.5,
            "Improvement\nRecommendations",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("💡 Recommendations", fontweight="bold", fontsize=12)
