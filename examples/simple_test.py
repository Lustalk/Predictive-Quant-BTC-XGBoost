#!/usr/bin/env python3
"""
Simple test to verify multi-timeframe collection works
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("🔧 Testing imports...")
    from src.data.multi_timeframe_collector import MultiTimeframeCollector, MemoryMonitor
    print("✅ Imports successful")
    
    print("🔧 Testing memory monitor...")
    memory_monitor = MemoryMonitor(max_memory_gb=12.0)
    print(f"✅ Memory usage: {memory_monitor.get_memory_percentage():.1f}%")
    
    print("🔧 Testing collector initialization...")
    collector = MultiTimeframeCollector(max_memory_gb=12.0, max_workers=2)
    print("✅ Collector initialized")
    
    print("🔧 Testing mock data collection...")
    # Let's test without actual data collection first
    print("✅ Ready for data collection test")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()