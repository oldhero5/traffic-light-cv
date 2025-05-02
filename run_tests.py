#!/usr/bin/env python3
"""
Run all tests for the traffic perception system.
"""
import argparse
import logging
import os
import sys
import unittest
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run traffic perception system tests")
    parser.add_argument("--test-dir", type=str, default="tests", help="Directory containing tests")
    parser.add_argument("--pattern", type=str, default="test_*.py", help="Test file pattern")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--module", type=str, help="Run tests for specific module (e.g., 'detector')")
    return parser.parse_args()


def run_tests(test_dir, pattern, verbose=False, module=None):
    # Prepare logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("test_runner")
    
    # Find test directory
    project_dir = Path(__file__).parent
    test_dir = project_dir / test_dir
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return 1
    
    # Adjust pattern for specific module
    if module:
        pattern = f"test_{module}.py"
    
    # Discover tests
    logger.info(f"Discovering tests in {test_dir} with pattern '{pattern}'")
    
    # Add project directory to path
    sys.path.insert(0, str(project_dir))
    
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
        module=args.module
    ))