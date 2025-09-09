#!/usr/bin/env python3
"""
Run all tests for the traffic perception system with M1 optimization support.
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run traffic perception system tests")
    parser.add_argument("--test-dir", type=str, default="tests", help="Directory containing tests")
    parser.add_argument("--pattern", type=str, default="test_*.py", help="Test file pattern")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--module", type=str, help="Run tests for specific module (e.g., 'detector')")
    parser.add_argument("--m1-only", action="store_true", help="Run only M1-specific tests")
    parser.add_argument("--skip-m1", action="store_true", help="Skip M1-specific tests")
    parser.add_argument("--performance", action="store_true", help="Run performance benchmark tests")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage reporting")
    parser.add_argument("--pytest", action="store_true", help="Use pytest instead of unittest")
    return parser.parse_args()


def check_m1_compatibility() -> dict[str, bool]:
    """Check M1 compatibility and available features."""
    compatibility = {
        'is_apple_silicon': platform.processor() == 'arm',
        'mps_available': torch.backends.mps.is_available(),
        'metal_available': hasattr(torch.backends, 'mps'),
    }
    
    # Check for CoreML tools
    try:
        import coremltools
        compatibility['coreml_available'] = True
    except ImportError:
        compatibility['coreml_available'] = False
    
    return compatibility


def run_pytest(args: list[str]) -> int:
    """Run pytest with specified arguments."""
    cmd = ['python', '-m', 'pytest'] + args
    return subprocess.call(cmd)


def run_tests(test_dir, pattern, verbose=False, module=None, use_pytest=False,
              m1_only=False, skip_m1=False, performance=False, coverage=False):
    """Run tests with enhanced M1 support."""
    
    # Check M1 compatibility first
    compat = check_m1_compatibility()
    
    # Prepare logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("test_runner")
    
    print("=== TrafficVision AI Test Runner ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Apple Silicon: {compat['is_apple_silicon']}")
    print(f"MPS Available: {compat['mps_available']}")
    print(f"CoreML Available: {compat['coreml_available']}")
    print("=" * 40)
    
    # Find test directory
    project_dir = Path(__file__).parent
    test_dir = project_dir / test_dir
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return 1
    
    # Add project directory to path
    sys.path.insert(0, str(project_dir))
    
    if use_pytest or m1_only or skip_m1 or performance or coverage:
        # Use pytest for advanced features
        pytest_args = []
        
        if verbose:
            pytest_args.extend(['-v', '-s'])
        
        if coverage:
            pytest_args.extend([
                '--cov=src',
                '--cov-report=term',
                '--cov-report=html:htmlcov',
                '--cov-fail-under=80'
            ])
        
        if m1_only:
            if not compat['mps_available']:
                print("❌ Error: M1-only tests requested but MPS not available")
                return 1
            pytest_args.extend(['-m', 'm1_required'])
        elif skip_m1:
            pytest_args.extend(['-m', 'not m1_required'])
        
        if performance:
            pytest_args.extend(['-m', 'performance'])
        
        if module:
            pytest_args.append(f"tests/test_{module}.py")
        else:
            pytest_args.append(str(test_dir))
        
        logger.info(f"Running pytest with arguments: {' '.join(pytest_args)}")
        exit_code = run_pytest(pytest_args)
        
        if exit_code == 0:
            print("✅ All tests passed!")
            
            # Run M1 validation if on Apple Silicon
            if compat['is_apple_silicon'] and not skip_m1:
                print("\n=== M1 Validation Report ===")
                validation_args = [
                    'tests/test_deepsort_tracker.py::TestDeepSORTTracker::test_validate_30_frame_tracking',
                    '-v'
                ]
                validation_code = run_pytest(validation_args)
                
                if validation_code == 0:
                    print("✅ M1 performance requirements validated!")
                else:
                    print("⚠️  M1 performance validation had issues")
        else:
            print(f"❌ Tests failed with exit code: {exit_code}")
        
        return exit_code
    
    else:
        # Use unittest for basic functionality
        # Adjust pattern for specific module
        if module:
            pattern = f"test_{module}.py"
        
        # Discover tests
        logger.info(f"Discovering tests in {test_dir} with pattern '{pattern}'")
        
        # Discover and run tests
        loader = unittest.TestLoader()
        tests = loader.discover(str(test_dir), pattern=pattern)
        
        # Create test runner
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        
        # Run tests
        logger.info("Running tests...")
        result = runner.run(tests)
        
        # Report results
        logger.info(f"Tests ran: {result.testsRun}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")
        
        # Return exit code
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    args = parse_args()
    
    sys.exit(run_tests(
        args.test_dir,
        args.pattern,
        verbose=args.verbose,
        module=args.module,
        use_pytest=args.pytest,
        m1_only=args.m1_only,
        skip_m1=args.skip_m1,
        performance=args.performance,
        coverage=args.coverage
    ))