# PyTorch ML Graph Optimizer

**Optimize PyTorch models with compiler-level techniques.**

This project demonstrates graph-level optimizations for deep learning models using PyTorch. It fuses convolution + batch normalization (+ optional ReLU), removes redundant layers, profiles latency, and generates visual summaries of the optimization effect.

---

## Features

- **Conv+BN(+ReLU) fusion** for faster inference
- **Remove nn.Identity layers** for cleaner models
- **Latency profiling** on CPU or GPU
- **Visual output**:
  - Layer composition before vs after optimization
  - Inference latency comparison
- **TorchScript export** for compiler-ready models

