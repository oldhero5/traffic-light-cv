#!/usr/bin/env python3
# scripts/hooks/validate_performance.py

import sys
import platform
import json
from pathlib import Path

def validate_performance():
    """Validate performance metrics against M1 targets."""
    
    # Skip if not on M1 Mac
    if not (platform.machine() == "arm64" and platform.system() == "Darwin"):
        print("⚠️  Not on M1 Mac - skipping performance validation")
        return 0
    
    # Look for performance metrics file
    metrics_file = Path("performance_metrics.json")
    if not metrics_file.exists():
        print("⚠️  No performance metrics found - create performance_metrics.json")
        return 0
    
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        errors = []
        
        # Define M1 performance targets
        targets = {
            'detection_fps_4k': 60,
            'detection_fps_1080p': 120,
            'tracking_fps': 60,
            'memory_usage_mb': 2000,
            'power_consumption_watts': 10,
            'latency_ms': 50
        }
        
        # Check each metric
        for metric, target in targets.items():
            if metric in metrics:
                value = metrics[metric]
                if metric in ['detection_fps_4k', 'detection_fps_1080p', 'tracking_fps']:
                    # Higher is better for FPS
                    if value < target:
                        errors.append(f"{metric}: {value} < {target} (target)")
                elif metric in ['memory_usage_mb', 'power_consumption_watts', 'latency_ms']:
                    # Lower is better for these metrics
                    if value > target:
                        errors.append(f"{metric}: {value} > {target} (target)")
        
        if errors:
            print("❌ Performance validation failed:")
            for error in errors:
                print(f"   {error}")
            return 1
        
        print("✅ Performance validation passed")
        return 0
        
    except Exception as e:
        print(f"⚠️  Could not validate performance metrics: {e}")
        return 0

if __name__ == "__main__":
    sys.exit(validate_performance())