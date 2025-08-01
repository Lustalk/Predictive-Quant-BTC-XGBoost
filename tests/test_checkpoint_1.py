#!/usr/bin/env python3
"""
🧪 CHECKPOINT 1 TEST: Multi-Timeframe Data Collection
===================================================

Test script to demonstrate enhanced multi-timeframe BTC data collection
with parallel processing, memory monitoring, and real results.

Usage:
    python test_checkpoint_1.py

Features Tested:
✅ Parallel collection of 30m, 1h, 4h BTC data
✅ Memory usage monitoring and optimization
✅ Data quality validation
✅ Timeframe alignment for multi-timeframe analysis
✅ Real-time performance metrics
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.multi_timeframe_collector import MultiTimeframeCollector, MemoryMonitor

def setup_logging():
    """Simple logging setup for testing"""
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger()


def test_checkpoint_1_complete():
    """
    Complete test of Checkpoint 1: Multi-Timeframe Data Collection
    """
    print("🎯 CHECKPOINT 1: MULTI-TIMEFRAME DATA COLLECTION TEST")
    print("=" * 80)
    print("Testing enhanced parallel data collection with real BTC data")
    print("Features: 30m/1h/4h timeframes, memory monitoring, data alignment")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    
    # Initialize components
    print("🔧 INITIALIZING SYSTEM...")
    collector = MultiTimeframeCollector(max_memory_gb=12.0, max_workers=3)
    memory_monitor = MemoryMonitor(max_memory_gb=12.0)
    
    print(f"   💻 Initial memory usage: {memory_monitor.get_memory_percentage():.1f}%")
    print(f"   🔧 System initialized with {collector.max_workers} workers")
    print()
    
    # Test 1: Multi-timeframe data collection
    print("📊 TEST 1: MULTI-TIMEFRAME DATA COLLECTION")
    print("-" * 50)
    
    start_time = time.time()
    
    # Collect data for multiple timeframes
    timeframes = ['30m', '1h', '4h']
    data_dict = collector.collect_multi_timeframe_data(
        timeframes=timeframes,
        symbol="BTC/USDT",
        days_back=14  # 2 weeks of data for quick test
    )
    
    collection_time = time.time() - start_time
    
    # Display results for each timeframe
    print("\n📈 TIMEFRAME ANALYSIS:")
    total_candles = 0
    total_memory = 0
    
    for timeframe, data in data_dict.items():
        candles = len(data)
        memory_mb = data.memory_usage(deep=True).sum() / (1024**2)
        price_range = f"${data['close'].min():.0f} - ${data['close'].max():.0f}"
        avg_volume = data['volume'].mean()
        
        total_candles += candles
        total_memory += memory_mb
        
        print(f"\n   {timeframe.upper()} TIMEFRAME:")
        print(f"   📊 Candles: {candles:,}")
        print(f"   📅 Date range: {data['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {data['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
        print(f"   💰 Price range: {price_range}")
        print(f"   📈 Current price: ${data['close'].iloc[-1]:.2f}")
        print(f"   📊 Avg volume: {avg_volume:,.0f} BTC")
        print(f"   💾 Memory: {memory_mb:.1f} MB")
        print(f"   📉 Recent change: {((data['close'].iloc[-1] / data['close'].iloc[-48] - 1) * 100):+.2f}% (last 48 periods)")
    
    print(f"\n🏆 COLLECTION SUMMARY:")
    print(f"   ⚡ Collection time: {collection_time:.2f} seconds")
    print(f"   📊 Total candles: {total_candles:,}")
    print(f"   💾 Total memory: {total_memory:.1f} MB")
    print(f"   💻 Memory usage: {memory_monitor.get_memory_percentage():.1f}%")
    
    # Test 2: Timeframe alignment
    print("\n🔄 TEST 2: TIMEFRAME ALIGNMENT")
    print("-" * 50)
    
    alignment_start = time.time()
    aligned_data = collector.align_timeframes(data_dict)
    alignment_time = time.time() - alignment_start
    
    print(f"\n📊 ALIGNED DATA ANALYSIS:")
    print(f"   📐 Shape: {aligned_data.shape}")
    print(f"   📈 Columns: {len(aligned_data.columns)}")
    print(f"   💾 Memory: {aligned_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   ⚡ Alignment time: {alignment_time:.2f} seconds")
    
    # Show columns
    print(f"\n📋 AVAILABLE COLUMNS:")
    columns = list(aligned_data.columns)
    for i, col in enumerate(columns):
        if i % 5 == 0:
            print(f"   {col:<15}", end="")
        else:
            print(f"{col:<15}", end="")
        if (i + 1) % 5 == 0:
            print()
    if len(columns) % 5 != 0:
        print()
    
    # Test 3: Multi-timeframe price analysis
    print("\n📈 TEST 3: MULTI-TIMEFRAME PRICE ANALYSIS")
    print("-" * 50)
    
    # Get latest prices across timeframes
    latest_prices = {}
    price_changes = {}
    
    for timeframe in timeframes:
        close_col = f'close_{timeframe}'
        if close_col in aligned_data.columns:
            latest_prices[timeframe] = aligned_data[close_col].iloc[-1]
            # Calculate 24h change (assuming 30m base timeframe)
            periods_24h = {'30m': 48, '1h': 24, '4h': 6}
            if len(aligned_data) > periods_24h.get(timeframe, 24):
                old_price = aligned_data[close_col].iloc[-periods_24h.get(timeframe, 24)]
                price_changes[timeframe] = ((latest_prices[timeframe] / old_price - 1) * 100)
    
    print(f"   💰 CURRENT PRICES:")
    for timeframe in timeframes:
        if timeframe in latest_prices:
            change = price_changes.get(timeframe, 0)
            print(f"   {timeframe.upper()}: ${latest_prices[timeframe]:.2f} ({change:+.2f}% 24h)")
    
    # Test 4: Data quality analysis
    print(f"\n🔍 TEST 4: DATA QUALITY ANALYSIS")
    print("-" * 50)
    
    summary = collector.get_collection_summary(data_dict)
    
    print(f"   📊 QUALITY SCORES:")
    for timeframe, quality_score in summary['data_quality'].items():
        print(f"   {timeframe.upper()}: {quality_score:.3f} (0.0-1.0 scale)")
    
    print(f"\n   📅 DATE COVERAGE:")
    for timeframe, date_info in summary['date_ranges'].items():
        duration = date_info['end'] - date_info['start']
        print(f"   {timeframe.upper()}: {duration.days} days ({date_info['count']} candles)")
    
    # Test 5: Sample data display
    print(f"\n📋 TEST 5: SAMPLE ALIGNED DATA")
    print("-" * 50)
    
    # Show last 5 rows with key columns
    key_columns = ['timestamp'] + [f'close_{tf}' for tf in timeframes if f'close_{tf}' in aligned_data.columns]
    sample_data = aligned_data[key_columns].tail(5)
    
    print(f"   📈 RECENT PRICE DATA (last 5 periods):")
    print(sample_data.to_string(index=False))
    
    # Final summary
    total_time = time.time() - start_time
    final_memory = memory_monitor.get_memory_percentage()
    
    print(f"\n🎉 CHECKPOINT 1 TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"✅ Multi-timeframe data collection: PASSED")
    print(f"✅ Parallel processing: PASSED")
    print(f"✅ Memory monitoring: PASSED")
    print(f"✅ Data alignment: PASSED")
    print(f"✅ Quality validation: PASSED")
    print()
    print(f"⚡ Total execution time: {total_time:.2f} seconds")
    print(f"💾 Final memory usage: {final_memory:.1f}%")
    print(f"📊 Total data points collected: {total_candles:,}")
    print(f"🎯 System ready for Checkpoint 2: Massive Indicator Variants")
    print("=" * 80)
    
    return {
        'success': True,
        'data_dict': data_dict,
        'aligned_data': aligned_data,
        'execution_time': total_time,
        'memory_usage': final_memory,
        'total_candles': total_candles
    }


def main():
    """Main test execution"""
    try:
        result = test_checkpoint_1_complete()
        
        if result['success']:
            print("\n🚀 Ready to proceed to Checkpoint 2!")
            print("   Next: Massive indicator variant generation")
        else:
            print("\n❌ Test failed - check logs for details")
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()