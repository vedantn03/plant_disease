"""
verify_gpu.py
=============
Quick sanity-check: confirms PyTorch sees the CUDA GPU.
Run: python scripts/verify_gpu.py
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def main() -> None:
    print(f"Python:  {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Check driver / CUDA wheel installation.")
        sys.exit(1)

    print(f"CUDA version (runtime): {torch.version.cuda}")
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Quick tensor test
    x = torch.randn(3, 224, 224, device="cuda")
    y = x * 2
    print(f"Tensor test: shape={y.shape}, device={y.device}")
    print("\nGPU verification PASSED ✓")


if __name__ == "__main__":
    main()
