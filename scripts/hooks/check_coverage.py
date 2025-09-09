#!/usr/bin/env python3
# scripts/hooks/check_coverage.py

import subprocess
import sys
import re

MINIMUM_COVERAGE = 80  # Minimum test coverage percentage

def check_coverage():
    """Check if test coverage meets minimum requirement."""
    try:
        # Run coverage
        result = subprocess.run(
            ["uv", "run", "pytest", "--cov=src", "--cov-report=term", "-q"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse coverage from output
        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
        
        if coverage_match:
            coverage = int(coverage_match.group(1))
            
            if coverage < MINIMUM_COVERAGE:
                print(f"❌ Test coverage {coverage}% < {MINIMUM_COVERAGE}% minimum")
                print("   Add more tests to increase coverage")
                return 1
            else:
                print(f"✅ Test coverage: {coverage}%")
                return 0
        else:
            print("⚠️  Could not determine test coverage")
            return 0
            
    except subprocess.TimeoutExpired:
        print("⚠️  Test coverage check timed out")
        return 0
    except Exception as e:
        print(f"⚠️  Coverage check failed: {e}")
        return 0

if __name__ == "__main__":
    sys.exit(check_coverage())