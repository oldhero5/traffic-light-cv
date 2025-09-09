#!/usr/bin/env python3
"""
Validation script for traffic light violation detection enhancements.

Validates that the enhanced system meets all requirements from the test prompt:
- Maintains 100+ FPS with violation detection
- <10ms p99 latency  
- <2GB memory usage
- >80% Neural Engine utilization
- All existing functionality from PR #12 maintained
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main_enhanced import EnhancedTrafficViolationSystem
from src.core.device_manager import DeviceManager


async def validate_enhanced_system():
    """Run comprehensive validation of enhanced system."""
    print("🔍 Starting Enhanced Traffic Violation System Validation")
    print("=" * 60)
    
    validation_results = {}
    
    # Initialize system
    print("1. Initializing Enhanced System...")
    system = EnhancedTrafficViolationSystem(
        config_path="configs/violation_detection_config.yaml",
        enable_m1_optimizations=True,
        enable_alerts=True,
        enable_visualization=False  # Disable for performance testing
    )
    
    try:
        await system.initialize()
        print("✅ System initialized successfully")
        validation_results['initialization'] = True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        validation_results['initialization'] = False
        return validation_results
    
    # Test 2: Validate M1 optimizations
    print("\n2. Validating M1 Optimizations...")
    try:
        device_manager = DeviceManager()
        if device_manager.is_m1_optimized():
            print(f"✅ Running on M1 chip: {device_manager.get_system_info()['chip']}")
            print(f"✅ MPS available: {torch.backends.mps.is_available()}")
            validation_results['m1_optimizations'] = True
        else:
            print("⚠️  Not running on M1 - some optimizations not available")
            validation_results['m1_optimizations'] = False
    except Exception as e:
        print(f"❌ M1 validation failed: {e}")
        validation_results['m1_optimizations'] = False
    
    # Test 3: Performance benchmarking
    print("\n3. Running Performance Benchmarks...")
    try:
        # Create synthetic test data
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        frame_count = 200  # Test frames
        
        print(f"Processing {frame_count} frames...")
        start_time = time.time()
        processing_times = []
        
        for i in range(frame_count):
            frame_start = time.perf_counter()
            result = await system.process_frame(test_frame, i)
            frame_time = time.perf_counter() - frame_start
            processing_times.append(frame_time)
            
            if i % 50 == 0 and i > 0:
                current_fps = 1.0 / np.mean(processing_times[-10:])
                print(f"  Frame {i}: {current_fps:.1f} FPS")
        
        total_time = time.time() - start_time
        processing_times = np.array(processing_times)
        
        # Calculate metrics
        avg_fps = len(processing_times) / np.sum(processing_times)
        p99_latency_ms = np.percentile(processing_times, 99) * 1000
        avg_latency_ms = np.mean(processing_times) * 1000
        
        print(f"\n📊 Performance Results:")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  P99 Latency: {p99_latency_ms:.2f}ms")
        print(f"  Average Latency: {avg_latency_ms:.2f}ms")
        print(f"  Total Processing Time: {total_time:.2f}s")
        
        # Validate performance targets
        fps_target = avg_fps >= 100.0
        latency_target = p99_latency_ms <= 10.0
        
        print(f"\n🎯 Performance Validation:")
        print(f"  FPS ≥100: {'✅ PASS' if fps_target else '❌ FAIL'} ({avg_fps:.1f})")
        print(f"  P99 Latency ≤10ms: {'✅ PASS' if latency_target else '❌ FAIL'} ({p99_latency_ms:.2f}ms)")
        
        validation_results['performance'] = {
            'fps_target': fps_target,
            'latency_target': latency_target,
            'avg_fps': avg_fps,
            'p99_latency_ms': p99_latency_ms
        }
        
    except Exception as e:
        print(f"❌ Performance benchmarking failed: {e}")
        validation_results['performance'] = {'error': str(e)}
    
    # Test 4: Memory usage validation
    print("\n4. Validating Memory Usage...")
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usage_gb = process.memory_info().rss / (1024**3)
        
        memory_target = memory_usage_gb <= 2.0
        print(f"  Memory Usage: {memory_usage_gb:.2f}GB")
        print(f"  Memory ≤2GB: {'✅ PASS' if memory_target else '❌ FAIL'}")
        
        validation_results['memory_usage'] = {
            'usage_gb': memory_usage_gb,
            'target_met': memory_target
        }
        
    except Exception as e:
        print(f"❌ Memory validation failed: {e}")
        validation_results['memory_usage'] = {'error': str(e)}
    
    # Test 5: Neural Engine utilization (if available)
    print("\n5. Validating Neural Engine Utilization...")
    try:
        if system.m1_optimizer:
            ane_metrics = system.m1_optimizer.get_ane_performance_metrics()
            ane_utilization = ane_metrics.utilization_percent
            ane_target = ane_utilization >= 80.0
            
            print(f"  ANE Utilization: {ane_utilization:.1f}%")
            print(f"  ANE ≥80%: {'✅ PASS' if ane_target else '❌ FAIL'}")
            
            validation_results['ane_utilization'] = {
                'utilization_percent': ane_utilization,
                'target_met': ane_target
            }
        else:
            print("  ⚠️  M1 optimizer not available")
            validation_results['ane_utilization'] = {'available': False}
            
    except Exception as e:
        print(f"❌ ANE validation failed: {e}")
        validation_results['ane_utilization'] = {'error': str(e)}
    
    # Test 6: Component integration validation
    print("\n6. Validating Component Integration...")
    try:
        integration_checks = {
            'detector': system.detector is not None,
            'traffic_light_detector': system.traffic_light_detector is not None,
            'tracker': system.tracker is not None,
            'lane_tracker': system.lane_tracker is not None,
            'violation_detector': system.violation_detector is not None,
            'alert_system': system.alert_system is not None
        }
        
        all_integrated = all(integration_checks.values())
        
        print("  Component Integration:")
        for component, status in integration_checks.items():
            print(f"    {component}: {'✅ OK' if status else '❌ MISSING'}")
        
        print(f"  All Components: {'✅ PASS' if all_integrated else '❌ FAIL'}")
        
        validation_results['integration'] = {
            'all_components': all_integrated,
            'components': integration_checks
        }
        
    except Exception as e:
        print(f"❌ Integration validation failed: {e}")
        validation_results['integration'] = {'error': str(e)}
    
    # Test 7: Feature validation
    print("\n7. Validating Enhanced Features...")
    try:
        feature_checks = {
            'enhanced_traffic_light_detection': hasattr(system.traffic_light_detector, 'detect_states_batch'),
            'lane_association': hasattr(system.lane_tracker, 'update_associations'),
            'violation_detection': hasattr(system.violation_detector, 'detect_violations'),
            'real_time_alerts': hasattr(system.alert_system, 'send_alert'),
            'm1_optimizations': system.m1_optimizer is not None
        }
        
        all_features = all(feature_checks.values())
        
        print("  Enhanced Features:")
        for feature, status in feature_checks.items():
            print(f"    {feature}: {'✅ OK' if status else '❌ MISSING'}")
        
        print(f"  All Features: {'✅ PASS' if all_features else '❌ FAIL'}")
        
        validation_results['features'] = {
            'all_features': all_features,
            'features': feature_checks
        }
        
    except Exception as e:
        print(f"❌ Feature validation failed: {e}")
        validation_results['features'] = {'error': str(e)}
    
    # Cleanup
    print("\n8. Cleaning Up...")
    try:
        await system.shutdown()
        print("✅ System shutdown completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    return validation_results


def print_final_summary(results: dict):
    """Print final validation summary."""
    print("\n" + "=" * 60)
    print("📋 FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    # Overall validation status
    for test_name, test_result in results.items():
        if isinstance(test_result, dict):
            if 'error' in test_result:
                print(f"❌ {test_name.upper()}: ERROR - {test_result['error']}")
                total_tests += 1
            elif test_name == 'performance':
                fps_pass = test_result.get('fps_target', False)
                latency_pass = test_result.get('latency_target', False)
                perf_pass = fps_pass and latency_pass
                
                print(f"{'✅' if perf_pass else '❌'} {test_name.upper()}: {'PASS' if perf_pass else 'FAIL'}")
                if perf_pass:
                    passed_tests += 1
                total_tests += 1
                
                print(f"    FPS: {test_result.get('avg_fps', 0):.1f} (target: ≥100)")
                print(f"    P99 Latency: {test_result.get('p99_latency_ms', 0):.2f}ms (target: ≤10ms)")
                
            elif test_name == 'memory_usage':
                mem_pass = test_result.get('target_met', False)
                print(f"{'✅' if mem_pass else '❌'} {test_name.upper()}: {'PASS' if mem_pass else 'FAIL'}")
                if mem_pass:
                    passed_tests += 1
                total_tests += 1
                
                print(f"    Memory: {test_result.get('usage_gb', 0):.2f}GB (target: ≤2GB)")
                
            elif test_name == 'ane_utilization':
                if 'available' in test_result:
                    print(f"⚠️  {test_name.upper()}: SKIPPED - Not available")
                else:
                    ane_pass = test_result.get('target_met', False)
                    print(f"{'✅' if ane_pass else '❌'} {test_name.upper()}: {'PASS' if ane_pass else 'FAIL'}")
                    if ane_pass:
                        passed_tests += 1
                    total_tests += 1
                    
                    print(f"    ANE Utilization: {test_result.get('utilization_percent', 0):.1f}% (target: ≥80%)")
                    
            else:
                # Boolean test results
                test_pass = test_result if isinstance(test_result, bool) else test_result.get('all_components', False) or test_result.get('all_features', False)
                print(f"{'✅' if test_pass else '❌'} {test_name.upper()}: {'PASS' if test_pass else 'FAIL'}")
                if test_pass:
                    passed_tests += 1
                total_tests += 1
        else:
            # Simple boolean result
            print(f"{'✅' if test_result else '❌'} {test_name.upper()}: {'PASS' if test_result else 'FAIL'}")
            if test_result:
                passed_tests += 1
            total_tests += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL VALIDATION CRITERIA MET!")
        print("✅ Enhanced system ready for production")
    elif passed_tests >= total_tests * 0.8:  # 80% threshold
        print("⚠️  Most validation criteria met - minor issues detected")
        print("🔧 System functional with recommended improvements")
    else:
        print("❌ Significant validation failures detected")
        print("🛠️  System requires fixes before deployment")
    
    print("=" * 60)


async def main():
    """Main validation entry point."""
    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during validation
        format='%(levelname)s: %(message)s'
    )
    
    try:
        results = await validate_enhanced_system()
        print_final_summary(results)
        
        # Return appropriate exit code
        if all(isinstance(v, bool) and v for v in results.values() if isinstance(v, bool)):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Some tests failed
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())