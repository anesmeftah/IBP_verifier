# ONNX Interval Bound Propagation (IBP) Verifier

A minimal Python implementation of Interval Bound Propagation (IBP) designed to compute output bounds for neural networks imported from ONNX models under $\mathcal{L}_\infty$ norm perturbations.

---

## Key Features

- **ONNX Parser**: Automatically extracts parameters (`Gemm`, `Relu`) from `.onnx` graph initializers.
- **Vectorized Interval Arithmetic**: Splits positive and negative weight components for efficient interval propagation.
- **Sound Bound Computation**: Computes exact lower and upper output bounds $[z_{\text{lower}}, z_{\text{upper}}]$ given an input $x$ and perturbation budget $\epsilon$.

---

## Mechanics

For a linear layer $y = Wx + b$, the interval bounds $[\underline{z}, \bar{z}]$ are propagated using:

$$W^+ = \max(W, 0), \quad W^- = \min(W, 0)$$

$$\bar{z} = W^+ \bar{x} + W^- \underline{x} + b$$
$$\underline{z} = W^+ \underline{x} + W^- \bar{x} + b$$

For ReLU activation layers:

$$\bar{z} = \max(0, \bar{x}), \quad \underline{z} = \max(0, \underline{x})$$

---

## Installation

Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install numpy onnx