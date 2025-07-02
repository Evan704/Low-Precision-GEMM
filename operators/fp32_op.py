# operators/fp16_op.py
import torch
from .base_op import OperatorWrapper

class PyTorchFP32GEMM(OperatorWrapper):
    def name(self) -> str:
        return "PyTorch-FP32-GEMM"

    def execute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)