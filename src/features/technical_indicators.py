"""
Technical indicators module for cryptocurrency trading.
Implements 20+ technical indicators without using ta-lib (as per requirements).
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union
import logging

from ..utils.logging import get_logger


class TechnicalIndicators:
    """
    Comprehensive technical indicators implementation.
    All indicators are implemented from scratch for full control and understanding.
    """
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        self.logger = get_logger().get_logger()
    
    def sma(self, data: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            data: Price series (typically close prices)
            window: Number of periods for the moving average
            
        Returns:
            Simple moving average series
        """
        return data.rolling(window=window, min_periods=1).mean()
    
    def ema(self, data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            data: Price series
            window: Number of periods
            alpha: Smoothing factor. If None, calculated as 2/(window+1)
            
        Returns:
            Exponential moving average series
        """
        if alpha is None:
            alpha = 2.0 / (window + 1.0)
        
        ema_values = np.zeros(len(data))
        ema_values[0] = data.iloc[0]
        
        for i in range(1, len(data)):
            ema_values[i] = alpha * data.iloc[i] + (1 - alpha) * ema_values[i-1]
        
        return pd.Series(ema_values, index=data.index, name=f'EMA_{window}')
    
    def rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            data: Price series
            window: RSI period
            
        Returns:
            RSI series (0-100)
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }, index=data.index)
    
    def bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands.
        
        Args:
            data: Price series
            window: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with Upper, Middle, and Lower bands
        """
        middle = self.sma(data, window)
        rolling_std = data.rolling(window=window, min_periods=1).std()
        
        upper = middle + (rolling_std * std_dev)
        lower = middle - (rolling_std * std_dev)
        
        return pd.DataFrame({
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower,
            'BB_Width': (upper - lower) / middle,  # Bandwidth
            'BB_Position': (data - lower) / (upper - lower)  # %B
        }, index=data.index)
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: ATR period
            
        Returns:
            ATR series
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=window, min_periods=1).mean()
        
        return atr
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_window: %K period
            d_window: %D smoothing period
            
        Returns:
            DataFrame with %K and %D
        """
        lowest_low = low.rolling(window=k_window, min_periods=1).min()
        highest_high = high.rolling(window=k_window, min_periods=1).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        }, index=close.index)
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Williams %R.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Williams %R period
            
        Returns:
            Williams %R series (-100 to 0)
        """
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Commodity Channel Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: CCI period
            
        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window, min_periods=1).mean()
        mad = typical_price.rolling(window=window, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def roc(self, data: pd.Series, window: int = 12) -> pd.Series:
        """
        Rate of Change.
        
        Args:
            data: Price series
            window: ROC period
            
        Returns:
            ROC series (percentage)
        """
        roc = ((data - data.shift(window)) / data.shift(window)) * 100
        return roc
    
    def momentum(self, data: pd.Series, window: int = 10) -> pd.Series:
        """
        Momentum indicator.
        
        Args:
            data: Price series
            window: Momentum period
            
        Returns:
            Momentum series
        """
        return data - data.shift(window)
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume.
        
        Args:
            close: Close prices
            volume: Volume series
            
        Returns:
            OBV series
        """
        price_change = close.diff()
        obv_values = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv_values[i] = obv_values[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv_values[i] = obv_values[i-1] - volume.iloc[i]
            else:
                obv_values[i] = obv_values[i-1]
        
        return pd.Series(obv_values, index=close.index, name='OBV')
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
        """
        Average Directional Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: ADX period
            
        Returns:
            DataFrame with ADX, +DI, -DI
        """
        # Calculate True Range
        atr_values = self.atr(high, low, close, window)
        
        # Calculate directional movements
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Smooth the directional movements
        plus_di = 100 * (plus_dm.rolling(window=window, min_periods=1).mean() / atr_values)
        minus_di = 100 * (minus_dm.rolling(window=window, min_periods=1).mean() / atr_values)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window, min_periods=1).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di
        }, index=close.index)
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume series
            
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    def ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series,
                 tenkan_window: int = 9, kijun_window: int = 26, 
                 senkou_b_window: int = 52) -> pd.DataFrame:
        """
        Ichimoku Cloud components.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            tenkan_window: Tenkan-sen period
            kijun_window: Kijun-sen period
            senkou_b_window: Senkou Span B period
            
        Returns:
            DataFrame with Ichimoku components
        """
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_window, min_periods=1).max()
        tenkan_low = low.rolling(window=tenkan_window, min_periods=1).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_window, min_periods=1).max()
        kijun_low = low.rolling(window=kijun_window, min_periods=1).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=senkou_b_window, min_periods=1).max()
        senkou_b_low = low.rolling(window=senkou_b_window, min_periods=1).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_window)
        
        # Chikou Span (Lagging Span)
        chikou = close.shift(-kijun_window)
        
        return pd.DataFrame({
            'Tenkan_sen': tenkan_sen,
            'Kijun_sen': kijun_sen,
            'Senkou_A': senkou_a,
            'Senkou_B': senkou_b,
            'Chikou': chikou
        }, index=close.index)
    
    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """
        Calculate daily pivot points.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            DataFrame with pivot levels
        """
        # Use previous day's HLC for pivot calculation
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Support and Resistance levels
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = r1 + (prev_high - prev_low)
        
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = s1 - (prev_high - prev_low)
        
        return pd.DataFrame({
            'Pivot': pivot,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }, index=close.index)
    
    def calculate_all_indicators(self, 
                               high: pd.Series, 
                               low: pd.Series, 
                               close: pd.Series, 
                               volume: pd.Series) -> pd.DataFrame:
        """
        Calculate all technical indicators at once.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume series
            
        Returns:
            DataFrame with all technical indicators
        """
        indicators = pd.DataFrame(index=close.index)
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            indicators[f'SMA_{window}'] = self.sma(close, window)
            indicators[f'EMA_{window}'] = self.ema(close, window)
        
        # Oscillators
        indicators['RSI_14'] = self.rsi(close, 14)
        indicators['RSI_30'] = self.rsi(close, 30)
        
        # MACD
        macd_data = self.macd(close)
        indicators = pd.concat([indicators, macd_data], axis=1)
        
        # Bollinger Bands
        bb_data = self.bollinger_bands(close)
        indicators = pd.concat([indicators, bb_data], axis=1)
        
        # ATR
        indicators['ATR_14'] = self.atr(high, low, close, 14)
        
        # Stochastic
        stoch_data = self.stochastic(high, low, close)
        indicators = pd.concat([indicators, stoch_data], axis=1)
        
        # Williams %R
        indicators['Williams_R'] = self.williams_r(high, low, close)
        
        # CCI
        indicators['CCI'] = self.cci(high, low, close)
        
        # Rate of Change
        indicators['ROC_12'] = self.roc(close, 12)
        
        # Momentum
        indicators['Momentum_10'] = self.momentum(close, 10)
        
        # Volume indicators
        indicators['OBV'] = self.obv(close, volume)
        indicators['VWAP'] = self.vwap(high, low, close, volume)
        
        # ADX
        adx_data = self.adx(high, low, close)
        indicators = pd.concat([indicators, adx_data], axis=1)
        
        # Ichimoku
        ichimoku_data = self.ichimoku(high, low, close)
        indicators = pd.concat([indicators, ichimoku_data], axis=1)
        
        # Pivot Points
        pivot_data = self.pivot_points(high, low, close)
        indicators = pd.concat([indicators, pivot_data], axis=1)
        
        self.logger.info(f"Calculated {len(indicators.columns)} technical indicators")
        return indicators