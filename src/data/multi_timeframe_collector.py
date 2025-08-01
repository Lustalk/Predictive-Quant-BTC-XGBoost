"""
Enhanced Multi-Timeframe Data Collection System
==============================================

Parallel collection and alignment of BTC data across multiple timeframes.
Optimized for 8-core systems with intelligent memory management.

Features:
- Parallel timeframe collection (30m, 1h, 4h)
- Memory usage monitoring
- Data quality validation
- Timeframe alignment for analysis
- Real-time progress tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
from dataclasses import dataclass

from .collectors import BinanceDataCollector
from ..utils.logging import get_logger


@dataclass
class DataCollectionStats:
    """Statistics for data collection performance"""
    timeframe: str
    candles_collected: int
    memory_usage_mb: float
    collection_time_seconds: float
    data_quality_score: float
    

class MemoryMonitor:
    """Monitor system memory usage during data collection"""
    
    def __init__(self, max_memory_gb: float = 12.0):
        self.max_memory_gb = max_memory_gb
        self.logger = get_logger().get_logger()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        return psutil.virtual_memory().used / (1024**3)
    
    def get_memory_percentage(self) -> float:
        """Get memory usage as percentage"""
        return psutil.virtual_memory().percent
    
    def check_memory_available(self, required_gb: float = 2.0) -> bool:
        """Check if enough memory is available"""
        available_gb = psutil.virtual_memory().available / (1024**3)
        return available_gb >= required_gb
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()
        self.logger.info("Forced garbage collection completed")


class MultiTimeframeCollector:
    """
    Enhanced data collector for multiple timeframes with parallel processing
    """
    
    def __init__(self, max_memory_gb: float = 12.0, max_workers: int = 8):
        self.logger = get_logger().get_logger()
        self.collector = BinanceDataCollector()
        self.memory_monitor = MemoryMonitor(max_memory_gb)
        self.max_workers = min(max_workers, 8)  # Limit to 8 cores
        
        self.logger.info(f"MultiTimeframeCollector initialized: {self.max_workers} workers, "
                        f"{max_memory_gb}GB memory limit")
    
    def collect_multi_timeframe_data(self, 
                                   timeframes: List[str] = ['30m', '1h', '4h'],
                                   symbol: str = "BTC/USDT",
                                   days_back: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple timeframes in parallel
        
        Args:
            timeframes: List of timeframes to collect
            symbol: Trading pair symbol
            days_back: Number of days to collect
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        print(f"📊 COLLECTING MULTI-TIMEFRAME DATA FOR {symbol}")
        print("=" * 60)
        print(f"   🎯 Timeframes: {timeframes}")
        print(f"   📅 Days back: {days_back}")
        print(f"   🔄 Workers: {self.max_workers}")
        print(f"   💾 Memory limit: {self.memory_monitor.max_memory_gb}GB")
        print("-" * 60)
        
        start_time = time.time()
        collection_stats = []
        
        def collect_single_timeframe(timeframe: str) -> Tuple[str, pd.DataFrame, DataCollectionStats]:
            """Collect data for a single timeframe"""
            tf_start_time = time.time()
            
            print(f"   📈 Starting {timeframe} collection...")
            
            # Check memory before collection
            if not self.memory_monitor.check_memory_available(1.0):
                self.memory_monitor.force_garbage_collection()
            
            try:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                # Collect data
                data = self.collector.collect_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                # Calculate statistics
                memory_usage = data.memory_usage(deep=True).sum() / (1024**2)
                collection_time = time.time() - tf_start_time
                quality_score = self._calculate_data_quality_score(data)
                
                stats = DataCollectionStats(
                    timeframe=timeframe,
                    candles_collected=len(data),
                    memory_usage_mb=memory_usage,
                    collection_time_seconds=collection_time,
                    data_quality_score=quality_score
                )
                
                print(f"   ✅ {timeframe}: {len(data)} candles, "
                      f"{memory_usage:.1f}MB, {collection_time:.1f}s")
                
                return timeframe, data, stats
                
            except Exception as e:
                print(f"   ❌ {timeframe} collection failed: {e}")
                self.logger.error(f"Failed to collect {timeframe} data: {e}")
                raise
        
        # Parallel collection using ThreadPoolExecutor
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all timeframe collection tasks
            future_to_timeframe = {
                executor.submit(collect_single_timeframe, tf): tf 
                for tf in timeframes
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_timeframe):
                timeframe = future_to_timeframe[future]
                try:
                    tf, data, stats = future.result()
                    results[tf] = data
                    collection_stats.append(stats)
                except Exception as e:
                    self.logger.error(f"Error collecting {timeframe}: {e}")
                    raise
        
        # Final statistics
        total_time = time.time() - start_time
        total_candles = sum(len(df) for df in results.values())
        total_memory = sum(df.memory_usage(deep=True).sum() for df in results.values()) / (1024**2)
        
        print("-" * 60)
        print("📊 COLLECTION SUMMARY:")
        print(f"   ✅ Total candles: {total_candles:,}")
        print(f"   💾 Total memory: {total_memory:.1f} MB")
        print(f"   ⚡ Total time: {total_time:.2f} seconds")
        print(f"   🎯 Avg quality score: {np.mean([s.data_quality_score for s in collection_stats]):.3f}")
        print(f"   💻 Memory usage: {self.memory_monitor.get_memory_percentage():.1f}%")
        print("=" * 60)
        
        return results
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-1)
        """
        score = 1.0
        
        # Penalize for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 0.3
        
        # Penalize for duplicate timestamps
        duplicate_ratio = data.duplicated(subset=['timestamp']).sum() / len(data)
        score -= duplicate_ratio * 0.2
        
        # Check OHLC consistency
        ohlc_consistent = (
            (data['high'] >= data['low']) &
            (data['high'] >= data['open']) &
            (data['high'] >= data['close']) &
            (data['low'] <= data['open']) &
            (data['low'] <= data['close'])
        ).mean()
        score *= ohlc_consistent
        
        return max(0.0, min(1.0, score))
    
    def align_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align multiple timeframes to create unified dataset
        
        Args:
            data_dict: Dictionary mapping timeframes to DataFrames
            
        Returns:
            Aligned DataFrame with multi-timeframe features
        """
        print("🔄 ALIGNING TIMEFRAMES")
        print("-" * 30)
        
        # Find the highest frequency timeframe as base
        timeframe_minutes = {
            '30m': 30, '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
        }
        
        base_timeframe = min(data_dict.keys(), 
                           key=lambda x: timeframe_minutes.get(x, 999999))
        
        print(f"   📊 Base timeframe: {base_timeframe}")
        
        # Start with base timeframe data
        aligned_data = data_dict[base_timeframe].copy()
        aligned_data.set_index('timestamp', inplace=True)
        
        # Rename columns to include timeframe suffix
        price_volume_cols = ['open', 'high', 'low', 'close', 'volume']
        aligned_data = aligned_data.rename(columns={
            col: f"{col}_{base_timeframe}" for col in price_volume_cols
        })
        
        # Align other timeframes
        for timeframe, data in data_dict.items():
            if timeframe == base_timeframe:
                continue
                
            print(f"   🔄 Aligning {timeframe}...")
            
            # Set timestamp as index for alignment
            tf_data = data.copy()
            tf_data.set_index('timestamp', inplace=True)
            
            # Rename columns with timeframe suffix
            tf_data = tf_data.rename(columns={
                col: f"{col}_{timeframe}" for col in price_volume_cols
            })
            
            # Forward fill to align with base timeframe
            aligned_data = aligned_data.join(tf_data, how='left')
            aligned_data[[f"{col}_{timeframe}" for col in price_volume_cols]] = \
                aligned_data[[f"{col}_{timeframe}" for col in price_volume_cols]].ffill()
        
        # Reset index to get timestamp as column
        aligned_data.reset_index(inplace=True)
        
        print(f"   ✅ Aligned data shape: {aligned_data.shape}")
        print(f"   📊 Columns: {list(aligned_data.columns)}")
        print("-" * 30)
        
        return aligned_data
    
    def get_collection_summary(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate comprehensive summary of collected data
        """
        summary = {
            'timeframes': list(data_dict.keys()),
            'total_candles': sum(len(df) for df in data_dict.values()),
            'memory_usage_mb': sum(df.memory_usage(deep=True).sum() for df in data_dict.values()) / (1024**2),
            'date_ranges': {},
            'data_quality': {}
        }
        
        for timeframe, data in data_dict.items():
            summary['date_ranges'][timeframe] = {
                'start': data['timestamp'].min(),
                'end': data['timestamp'].max(),
                'count': len(data)
            }
            summary['data_quality'][timeframe] = self._calculate_data_quality_score(data)
        
        return summary


# Test function to demonstrate real results
def test_multi_timeframe_collection():
    """
    Test function to demonstrate multi-timeframe data collection
    """
    print("🧪 TESTING MULTI-TIMEFRAME DATA COLLECTION")
    print("=" * 80)
    
    collector = MultiTimeframeCollector(max_memory_gb=12.0, max_workers=3)
    
    # Collect data for 3 timeframes
    timeframes = ['30m', '1h', '4h']
    data_dict = collector.collect_multi_timeframe_data(
        timeframes=timeframes,
        symbol="BTC/USDT",
        days_back=30  # Start with 30 days for quick test
    )
    
    # Show results
    print("\n📊 DATA COLLECTION RESULTS:")
    for timeframe, data in data_dict.items():
        print(f"\n{timeframe.upper()} TIMEFRAME:")
        print(f"  📈 Candles: {len(data):,}")
        print(f"  📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"  💰 Price range: ${data['close'].min():.0f} - ${data['close'].max():.0f}")
        print(f"  📊 Avg volume: {data['volume'].mean():,.0f}")
        print(f"  💾 Memory: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Test alignment
    aligned_data = collector.align_timeframes(data_dict)
    
    print(f"\n🔄 ALIGNED DATA:")
    print(f"  📊 Shape: {aligned_data.shape}")
    print(f"  📈 Columns: {len(aligned_data.columns)}")
    print(f"  💾 Memory: {aligned_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Show sample of aligned data
    print(f"\n📋 SAMPLE DATA (last 5 rows):")
    print(aligned_data.tail()[['timestamp', 'close_30m', 'close_1h', 'close_4h']].to_string())
    
    return data_dict, aligned_data


if __name__ == "__main__":
    # Run test if executed directly
    test_multi_timeframe_collection()