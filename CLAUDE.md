# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**TrafficVision AI** - Traffic Light & Camera Perception System optimized for Apple Silicon (M1/M2) MacBook Pro. This is a high-performance computer vision system with Neural Engine optimization, Metal Performance Shaders, and macOS-native integration.

## M1 Development Environment Setup

### Package Manager & Python
- **MANDATORY**: Use `uv` for all dependency management
- **Python Version**: 3.11+ (ARM64 native for M1)
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create environment: `uv venv --python 3.11`
- Install dependencies: `uv pip install -r requirements-m1.txt`

### Build & Test Commands
- Install M1 dependencies: `uv pip install -r requirements-m1.txt`
- Install development deps: `uv pip install -r requirements-dev.txt`
- Install package: `uv pip install -e .`
- Run all tests: `uv run python run_tests.py`
- Run M1-specific tests: `uv run pytest tests/ -m m1_required`
- Run performance tests: `uv run pytest tests/benchmarks/ --benchmark-only`
- Verbose test output: `uv run python run_tests.py --verbose`
- Training: `uv run python -m src.detection.train --config configs/detection_config.yaml`
- Check coverage: `uv run pytest --cov=src --cov-report=term`
- Run linting: `uv run ruff check --fix`
- Run formatting: `uv run ruff format`

## M1-Specific Architecture Requirements

### Device Detection & Optimization
```python
# MANDATORY: Always check for MPS availability
import torch

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # M1 GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# All models MUST use optimal device
device = get_device()
model = model.to(device)
```

### CoreML Integration (MANDATORY)
- All PyTorch models MUST have CoreML versions for Neural Engine
- Use `compute_units=ct.ComputeUnit.ALL` for M1 optimization
- Target Neural Engine with `convert_to="neuralnetwork"`
- Use INT8 quantization when possible
- Profile ANE usage with `os.environ['COREML_PROFILING'] = '1'`

### Metal Performance Shaders
- All image processing MUST use Metal shaders when available
- Implement zero-copy operations using unified memory
- Use half precision (float16) for M1 efficiency
- Create buffer pools to avoid allocations

## Performance Requirements (MANDATORY)

### M1 Base Targets
- **4K Processing**: ≥60 FPS
- **1080p Processing**: ≥120 FPS
- **Latency**: <100ms end-to-end
- **Memory Usage**: <2GB for 4K
- **Power Consumption**: <10W average

### M1 Pro/Max/Ultra Targets
- **4K Processing**: ≥120 FPS
- **8K Processing**: ≥60 FPS (M1 Max/Ultra)
- **Latency**: <50ms end-to-end
- **Neural Engine Usage**: >80%
- **Multi-stream**: 4+ concurrent streams

## Code Style & Standards

### Python Standards
- **Python Version**: 3.11+ (ARM64 native)
- **Formatting**: Ruff with 100 character line-length
- **Type Hints**: MANDATORY for all functions
- **Imports**: standard library, third-party, local modules
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Numpy-style with M1 performance metrics

### M1-Specific Code Rules
```python
# MANDATORY: Include M1 performance metrics in docstrings
def process_frame(frame: np.ndarray) -> List[Detection]:
    """
    Process video frame with traffic light detection.
    
    M1 Performance:
    - FPS: 120 @ 4K, 240 @ 1080p
    - ANE Usage: 85%
    - Power: 7.5W average
    - Memory: 800MB
    - Max Temperature: 65°C
    """
```

### Error Handling
- Use specific exceptions, never bare `except:`
- Handle MPS-specific errors with CPU fallback
- Include performance degradation warnings

## Project Structure (MANDATORY)

```
traffic-light-cv/
├── src/
│   ├── core/               # Core M1 pipeline
│   ├── metal/              # Metal shaders (MANDATORY)
│   ├── coreml/             # CoreML models (MANDATORY)
│   ├── mps/                # MPS acceleration
│   ├── ane/                # Neural Engine optimization
│   └── utils/              # Utilities
├── models/
│   ├── pytorch/            # Original models
│   ├── coreml/             # M1-optimized models
│   └── onnx/               # Intermediate format
├── shaders/                # Metal shader files (.metal)
├── macos/                  # Native macOS app (SwiftUI)
├── tests/
│   ├── performance/        # M1 performance tests
│   └── benchmarks/         # M1 benchmarks
└── requirements-m1.txt     # M1-locked dependencies
```

## Testing Requirements

### M1-Specific Tests (MANDATORY)
```python
@pytest.mark.m1_required
def test_4k_fps_requirement():
    """Must achieve 60+ FPS on 4K processing."""
    # Test implementation

@pytest.mark.m1_required  
def test_neural_engine_usage():
    """Must use >80% Neural Engine utilization."""
    # Test implementation
```

### Performance Benchmarks
- Run before each commit: `uv run pytest tests/benchmarks/`
- Compare against baseline: `--benchmark-compare`
- Generate reports: `--benchmark-histogram`

## Git Workflow

### Branch Naming
```bash
# M1-specific features
feature/m1-neural-engine-optimization
feature/m1-metal-shaders
perf/m1-4k-120fps

# Standard features
feature/123-add-deepsort-tracking  # Must include issue number
bugfix/456-fix-memory-leak
```

### Commit Messages (MANDATORY)
```bash
# M1-specific commits must include [M1] tag
feat(m1): add Neural Engine optimization [M1]
perf(m1): improve 4K processing to 120 FPS [M1]
fix(m1): resolve MPS memory leak [M1]

# Follow conventional commits
type(scope): description (#issue)
```

## Security & Deployment

### macOS-Specific
- **Code Signing**: MANDATORY for distribution
- **Notarization**: Required for outside App Store
- **Hardened Runtime**: MANDATORY
- **App Sandbox**: MANDATORY for App Store
- Universal Binary support (Intel + Apple Silicon)

### Pre-commit Hooks
- Security scanning with bandit, trufflehog
- M1 compatibility checks
- Model size validation (<50MB)
- Performance metric validation
- Test coverage check (≥80%)

## Monitoring & Profiling

### Required Metrics
```python
REQUIRED_METRICS = {
    'fps': {'min': 60, 'target': 120},
    'latency_ms': {'max': 50, 'target': 30},
    'ane_usage_percent': {'min': 80, 'target': 95},
    'memory_mb': {'max': 2000, 'target': 1000},
    'power_watts': {'max': 10, 'target': 5},
    'temperature_c': {'max': 80, 'warning': 70},
}
```

### Instruments Integration (MANDATORY)
```bash
# Profile Neural Engine usage
xcrun instruments -t "Neural Engine" -D trace.trace -l 10000 -w "TrafficVision.app"

# Profile Metal GPU usage  
xcrun instruments -t "Metal System Trace" -D metal.trace -l 10000 -w "TrafficVision.app"
```

## Review Checklist

Before submitting M1-related code:
- [ ] Runs on MPS (Metal Performance Shaders)
- [ ] Has CoreML model version for Neural Engine
- [ ] Achieves target FPS (60+ for 4K, 120+ for 1080p)
- [ ] Neural Engine utilization >80%
- [ ] Power consumption <10W average
- [ ] Memory efficient (<2GB for 4K)
- [ ] Universal binary builds successfully
- [ ] Instruments profiling completed
- [ ] No thermal throttling under load
- [ ] M1-specific tests pass
- [ ] Benchmarks meet performance targets
- [ ] Documentation includes M1 metrics
- [ ] Pre-commit hooks pass
- [ ] Code signed and ready for distribution