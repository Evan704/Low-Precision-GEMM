# operators/fp16_op.py
import torch
from .base_op import OperatorWrapper

class PyTorchFP16GEMM(OperatorWrapper):
    """
    A robust and simple FP16 matrix multiplication baseline.
    This is guaranteed to work on any modern GPU.
    """
    def name(self) -> str:
        return "PyTorch-FP16-GEMM"

    def execute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Cast inputs to Half-Precision (FP16) and perform matmul
        # The .float() at the end casts the result back to FP32 for consistency
        return torch.matmul(a.half(), b.half()).float()