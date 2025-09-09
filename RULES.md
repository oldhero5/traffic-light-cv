# Traffic Light & Camera Perception System
## Development Rules & Guidelines - M1 MacBook Pro

### 1. M1 Development Environment Setup

#### 1.1 Python Setup with UV on M1

```bash
# Install uv on M1 Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify ARM64 installation
uv --version
file $(which uv)  # Should show "arm64"

# Create project with M1-optimized Python
uv init traffic-vision --python 3.11

# Install M1-optimized dependencies
cd traffic-vision
uv pip install torch torchvision torchaudio  # Will install MPS-enabled versions
uv pip install coremltools
uv pip install tensorflow-macos
uv pip install tensorflow-metal

# Lock dependencies for M1
uv pip compile requirements.in -o requirements-m1.txt --platform macos_arm64

# Create development environment
uv venv --python 3.11
source .venv/bin/activate
```

#### 1.2 M1-Specific Project Structure

```
MANDATORY M1 Project Structure:
traffic-light-cv/
├── .python-version          # MUST specify 3.11+ (ARM64 native)
├── pyproject.toml           # MUST include M1-specific deps
├── requirements-m1.txt      # MUST be M1-locked with uv
├── requirements-dev.txt     # MUST include M1 profiling tools
├── src/
│   ├── core/               # Core processing
│   ├── metal/              # Metal shaders (MANDATORY for M1)
│   ├── coreml/             # CoreML models (MANDATORY)
│   ├── mps/                # MPS acceleration
│   └── ane/                # Neural Engine optimization
├── models/
│   ├── pytorch/            # Original PyTorch models
│   ├── coreml/             # Converted CoreML models
│   └── onnx/               # ONNX intermediate format
├── shaders/                 # Metal shader files (.metal)
├── macos/                   # macOS app (SwiftUI)
├── tests/
│   ├── performance/         # M1 performance tests
│   └── benchmarks/          # M1 benchmarks
└── profiling/               # Instruments templates
```

### 2. M1-Specific Code Standards

#### 2.1 MPS (Metal Performance Shaders) Usage

```python
# MANDATORY: Always check for MPS availability
import torch

def get_device() -> str:
    """Get optimal device for M1 Mac."""
    if torch.backends.mps.is_available():
        # M1 GPU available
        return "mps"
    elif torch.cuda.is_available():
        # Should not happen on M1, but check anyway
        return "cuda"
    else:
        return "cpu"

# MANDATORY: Use MPS for all tensor operations
device = get_device()
model = model.to(device)
tensor = tensor.to(device)

# MANDATORY: Handle MPS-specific limitations
try:
    # Some operations not supported on MPS yet
    result = complex_operation(tensor)
except NotImplementedError:
    # Fallback to CPU for unsupported ops
    tensor_cpu = tensor.cpu()
    result = complex_operation(tensor_cpu)
    result = result.to(device)
```

#### 2.2 CoreML Integration Rules

```python
# MANDATORY: All models must have CoreML versions
import coremltools as ct

class ModelConverter:
    """Convert models for M1 optimization."""
    
    @staticmethod
    def pytorch_to_coreml(model_path: str) -> ct.models.MLModel:
        """
        MANDATORY: Convert PyTorch to CoreML.
        
        Rules:
        1. Must target Neural Engine
        2. Must use INT8 quantization when possible
        3. Must profile ANE usage
        """
        model = torch.load(model_path)
        model.eval()
        
        # MANDATORY: Use ComputeUnit.ALL for M1
        coreml_model = ct.convert(
            model,
            compute_units=ct.ComputeUnit.ALL,  # Use all M1 compute units
            convert_to="neuralnetwork",  # For ANE compatibility
        )
        
        # MANDATORY: Optimize for ANE
        spec = coreml_model.get_spec()
        ct.models.neural_network.optimize_for_ane(spec)
        
        return ct.models.MLModel(spec)
```

#### 2.3 Metal Shader Rules

```metal
// MANDATORY: All image processing must use Metal shaders
#include <metal_stdlib>
using namespace metal;

// MANDATORY: Optimize for M1 tile-based rendering
kernel void process_frame [[ max_total_threads_per_threadgroup(1024) ]] (
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant ProcessParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // MANDATORY: Check bounds
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
        return;
    }
    
    // MANDATORY: Use half precision for M1 efficiency
    half4 pixel = half4(input.read(gid));
    
    // Process pixel
    pixel = process_pixel(pixel, params);
    
    // Write result
    output.write(float4(pixel), gid);
}
```

### 3. M1 Performance Rules

#### 3.1 Memory Management

```python
# MANDATORY: Use unified memory efficiently
class UnifiedMemoryRules:
    """Rules for M1 unified memory usage."""
    
    # RULE 1: Pre-allocate buffers
    def __init__(self):
        self.buffer_pool = self._create_buffer_pool()
    
    # RULE 2: Zero-copy operations
    def process_frame(self, frame: np.ndarray):
        # WRONG: Creating copies
        frame_copy = frame.copy()
        processed = process(frame_copy)
        
        # CORRECT: In-place operations
        process_inplace(frame)
        
    # RULE 3: Reuse buffers
    def get_buffer(self):
        # WRONG: Allocating new buffer each time
        return np.zeros((1080, 1920, 3))
        
        # CORRECT: Reuse from pool
        return self.buffer_pool.get()
```

#### 3.2 Neural Engine Optimization

```python
# MANDATORY: Profile Neural Engine usage
import os
os.environ['COREML_PROFILING'] = '1'  # Enable profiling

class ANEOptimizationRules:
    """Rules for Neural Engine optimization."""
    
    # RULE 1: Batch size must be 1 for ANE
    BATCH_SIZE = 1  # ANE doesn't support batching
    
    # RULE 2: Use supported operations only
    SUPPORTED_OPS = [
        'Conv2D', 'BatchNorm', 'ReLU', 'MaxPool',
        'Add', 'Concat', 'GlobalAvgPool'
    ]
    
    # RULE 3: Quantize to INT8
    def quantize_for_ane(self, model):
        """MANDATORY: Quantize models for ANE."""
        return ct.models.neural_network.quantization_utils.quantize_weights(
            model, 
            nbits=8,
            quantization_mode="linear"
        )
```

### 4. Testing Rules for M1

#### 4.1 Performance Testing Requirements

```python
# tests/test_m1_requirements.py
import pytest
import time
import numpy as np

class TestM1Requirements:
    """MANDATORY performance requirements for M1."""
    
    @pytest.mark.m1_required
    def test_4k_fps_requirement(self, m1_pipeline):
        """MANDATORY: Must achieve 60+ FPS on 4K."""
        frame_4k = np.random.uint8((2160, 3840, 3))
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            m1_pipeline.process(frame_4k)
            times.append(time.perf_counter() - start)
        
        fps = 1.0 / np.mean(times)
        assert fps >= 60, f"4K FPS {fps} below required 60"
    
    @pytest.mark.m1_required
    def test_neural_engine_usage(self, m1_pipeline):
        """MANDATORY: Must use >80% Neural Engine."""
        metrics = m1_pipeline.get_ane_metrics()
        assert metrics['ane_utilization'] >= 80
    
    @pytest.mark.m1_required
    def test_power_consumption(self, m1_pipeline):
        """MANDATORY: Must use <10W average power."""
        power = measure_power_consumption(duration=60)
        assert power['average_watts'] < 10
```

#### 4.2 Benchmarking Rules

```bash
# MANDATORY: Run benchmarks before each commit
uv run pytest tests/benchmarks/ --benchmark-only

# MANDATORY: Compare against baseline
uv run pytest tests/benchmarks/ --benchmark-compare

# MANDATORY: Generate performance report
uv run pytest tests/benchmarks/ --benchmark-histogram
```

### 5. Git Workflow for M1 Development

#### 5.1 Branch Naming for M1 Features

```bash
# M1-specific features
feature/m1-neural-engine-optimization
feature/m1-metal-shaders
feature/m1-unified-memory

# M1 performance improvements
perf/m1-4k-120fps
perf/m1-power-optimization
perf/m1-memory-efficiency
```

#### 5.2 Commit Message Format

```bash
# M1-specific commits must include [M1] tag
git commit -m "feat(m1): add Neural Engine optimization [M1]"
git commit -m "perf(m1): improve 4K processing to 120 FPS [M1]"
git commit -m "fix(m1): resolve MPS memory leak [M1]"
```

### 6. Deployment Rules for macOS

#### 6.1 Universal Binary Requirements

```bash
# MANDATORY: Build universal binary
uv run python setup.py bdist_wheel --universal

# MANDATORY: Test on both architectures
arch -x86_64 uv run pytest  # Intel
arch -arm64 uv run pytest   # M1

# MANDATORY: Sign and notarize
codesign --deep --force --sign "Developer ID" dist/TrafficVision.app
xcrun notarytool submit dist/TrafficVision.app --wait
```

#### 6.2 App Store Submission Rules

```yaml
# MANDATORY: Info.plist requirements
required_keys:
  LSMinimumSystemVersion: "12.0"  # macOS Monterey minimum
  LSArchitecturePriority:
    - arm64  # M1 first
    - x86_64  # Intel fallback
  NSCameraUsageDescription: "Process dashcam footage"
  NSPhotoLibraryUsageDescription: "Access video files"
```

### 7. Monitoring Rules

#### 7.1 M1-Specific Metrics

```python
# MANDATORY: Monitor these metrics
class M1Metrics:
    """Required metrics for M1 monitoring."""
    
    REQUIRED_METRICS = {
        'fps': {'min': 60, 'target': 120},
        'latency_ms': {'max': 50, 'target': 30},
        'ane_usage_percent': {'min': 80, 'target': 95},
        'memory_mb': {'max': 2000, 'target': 1000},
        'power_watts': {'max': 10, 'target': 5},
        'temperature_c': {'max': 80, 'warning': 70},
    }
    
    def validate_metrics(self, metrics: dict) -> bool:
        """MANDATORY: Validate all metrics meet requirements."""
        for metric, thresholds in self.REQUIRED_METRICS.items():
            value = metrics.get(metric)
            if 'min' in thresholds and value < thresholds['min']:
                raise PerformanceError(f"{metric} {value} below minimum {thresholds['min']}")
            if 'max' in thresholds and value > thresholds['max']:
                raise PerformanceError(f"{metric} {value} above maximum {thresholds['max']}")
        return True
```

### 8. Debug Rules for M1

#### 8.1 Instruments Integration

```python
# MANDATORY: Use Instruments for profiling
class InstrumentsProfiler:
    """Profile with Xcode Instruments."""
    
    @staticmethod
    def profile_neural_engine():
        """MANDATORY: Profile ANE usage."""
        os.system("""
            xcrun instruments -t "Neural Engine" \
                -D trace.trace \
                -l 10000 \
                -w "TrafficVision.app"
        """)
    
    @staticmethod
    def profile_metal():
        """MANDATORY: Profile Metal usage."""
        os.system("""
            xcrun instruments -t "Metal System Trace" \
                -D metal.trace \
                -l 10000 \
                -w "TrafficVision.app"
        """)
```

### 9. CI/CD Rules for M1

```yaml
# .github/workflows/m1-ci.yml
name: M1 CI/CD

on: [push, pull_request]

jobs:
  test-m1:
    runs-on: macos-latest  # M1 runners
    
    steps:
      - uses: actions/checkout@v3
      
      # MANDATORY: Verify M1 architecture
      - name: Verify M1
        run: |
          arch | grep arm64
          sysctl -n machdep.cpu.brand_string | grep "Apple M"
      
      # MANDATORY: Install uv
      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      # MANDATORY: Install dependencies with uv
      - name: Install dependencies
        run: |
          uv pip install -r requirements-m1.txt
          uv pip install -r requirements-dev.txt
      
      # MANDATORY: Run M1-specific tests
      - name: Run M1 tests
        run: |
          uv run pytest tests/ -m m1_required
          uv run pytest tests/benchmarks/ --benchmark-only
      
      # MANDATORY: Check performance
      - name: Validate performance
        run: |
          uv run python scripts/validate_m1_performance.py
```

### 10. Security Rules for macOS

```python
# MANDATORY: macOS security requirements
class MacOSSecurity:
    """Security rules for macOS deployment."""
    
    # MANDATORY: Code signing
    CODESIGN_REQUIREMENTS = {
        'identity': 'Developer ID Application',
        'options': ['--deep', '--force', '--options', 'runtime'],
        'entitlements': 'entitlements.plist'
    }
    
    # MANDATORY: Notarization
    NOTARIZATION_REQUIRED = True
    
    # MANDATORY: Hardened runtime
    HARDENED_RUNTIME = True
    
    # MANDATORY: App Sandbox
    SANDBOX_ENABLED = True
```

### 11. Documentation Rules

```markdown
## MANDATORY: M1-Specific Documentation

Every feature must include:
1. M1 performance benchmarks
2. Neural Engine utilization metrics
3. Power consumption data
4. Memory usage statistics
5. Thermal impact assessment

Example:
```python
def new_feature():
    """
    Process video with new algorithm.
    
    M1 Performance:
    - FPS: 120 @ 4K, 240 @ 1080p
    - ANE Usage: 85%
    - Power: 7.5W average
    - Memory: 800MB
    - Max Temperature: 65°C
    """
```
```

### 12. Review Checklist for M1

Before submitting any M1-related code:

- [ ] Runs on MPS (Metal Performance Shaders)
- [ ] Has CoreML model version
- [ ] Uses Neural Engine (>80% utilization)
- [ ] Achieves target FPS (60+ for 4K)
- [ ] Power consumption <10W
- [ ] Memory efficient (<2GB for 4K)
- [ ] Universal binary builds
- [ ] Instruments profiling complete
- [ ] No thermal throttling
- [ ] Battery life impact measured
- [ ] Code signed and notarized
- [ ] M1-specific tests pass
- [ ] Benchmarks meet targets
- [ ] Documentation includes M1 metrics

### 13. Optimization Priority

```python
# MANDATORY: Optimization priority for M1
class OptimizationPriority:
    """Priority order for M1 optimizations."""
    
    PRIORITY_ORDER = [
        "Neural Engine",      # First: Use ANE for ML inference
        "Metal Shaders",      # Second: GPU for image processing  
        "MPS Operations",     # Third: PyTorch MPS backend
        "Unified Memory",     # Fourth: Zero-copy operations
        "SIMD/Neon",         # Fifth: CPU vectorization
        "Standard CPU",       # Last resort
    ]
    
    @classmethod
    def get_optimal_path(cls, operation: str) -> str:
        """Get optimal execution path for operation."""
        if operation in ML_OPERATIONS:
            return "Neural Engine"
        elif operation in IMAGE_OPERATIONS:
            return "Metal Shaders"
        elif operation in TENSOR_OPERATIONS:
            return "MPS Operations"
        else:
            return "SIMD/Neon"
```