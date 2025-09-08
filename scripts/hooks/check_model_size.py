#!/usr/bin/env python3
# scripts/hooks/check_model_size.py

import sys
import os
from pathlib import Path

MAX_MODEL_SIZE_MB = 50  # Maximum size for mobile deployment

def check_model_size():
    """Check if model files are within size limits."""
    model_extensions = ['.mlmodel', '.onnx', '.pt', '.pth', '.h5', '.pb']
    
    errors = []
    
    # Find all model files
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                file_path = Path(root) / file
                size_mb = file_path.stat().st_size / (1024 * 1024)
                
                if size_mb > MAX_MODEL_SIZE_MB:
                    errors.append(f"{file_path}: {size_mb:.1f}MB > {MAX_MODEL_SIZE_MB}MB")
                else:
                    print(f"✅ {file}: {size_mb:.1f}MB")
    
    if errors:
        print("\n❌ Model size check failed:")
        for error in errors:
            print(f"   {error}")
        print(f"\nModels must be <{MAX_MODEL_SIZE_MB}MB for mobile deployment")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(check_model_size())