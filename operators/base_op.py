# operators/base_op.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class BenchmarkConfig:
    """定义一个测试场景的配置"""
    m: int
    n: int
    k: int
    batch_size: int

class OperatorWrapper(ABC):
    """
    所有待测算子的包装器基类 (Interface)
    任何新的算子实现都需要继承此类，并实现其抽象方法。
    """

    @abstractmethod
    def name(self) -> str:
        """返回算子的名称，用于报告。"""
        pass

    def prepare(self, config: BenchmarkConfig, device: torch.device):
        """
        （可选）在正式执行前，进行一些准备工作。
        例如，对于某些实现（如CUTLASS），可能需要根据配置提前编译或分配工作空间。
        默认情况下什么都不做。
        """
        pass

    @abstractmethod
    def execute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        执行算子的核心逻辑。
        
        Args:
            a (torch.Tensor): 输入矩阵A，形状为 (B, M, K)
            b (torch.Tensor): 输入矩阵B，形状为 (B, K, N)

        Returns:
            torch.Tensor: 输出矩阵C，形状为 (B, M, N)
        """
        pass