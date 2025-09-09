#!/usr/bin/env python3
# scripts/hooks/check_m1_compatibility.py

import sys
import platform
import subprocess
import importlib.util

def check_m1_compatibility():
    """Verify code is M1-compatible."""
    errors = []
    warnings = []
    
    # Check if running on M1
    is_m1 = platform.machine() == "arm64" and platform.system() == "Darwin"
    
    if not is_m1:
        warnings.append("Not running on M1 Mac - skipping some checks")
    
    # Check for MPS availability if on Mac
    if platform.system() == "Darwin":
        try:
            import torch
            if not torch.backends.mps.is_available() and is_m1:
                warnings.append("MPS not available on M1 - check PyTorch installation")
        except ImportError:
            errors.append("PyTorch not installed")
    
    # Check for CoreML tools
    if importlib.util.find_spec("coremltools") is None:
        errors.append("coremltools not installed - required for M1 optimization")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 11):
        errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.11")
    
    # Report results
    if warnings:
        for warning in warnings:
            print(f"⚠️  Warning: {warning}")
    
    if errors:
        for error in errors:
            print(f"❌ Error: {error}")
        return 1
    
    print("✅ M1 compatibility check passed")
    return 0

if __name__ == "__main__":
    sys.exit(check_m1_compatibility())