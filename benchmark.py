# benchmark.py

import torch
import pandas as pd
from typing import List
from tqdm import tqdm
from operators.base_op import OperatorWrapper, BenchmarkConfig

class BenchmarkRunner:
    def __init__(self, 
                 configs: List[BenchmarkConfig], 
                 operators: List[OperatorWrapper], 
                 n_warmup: int = 10, 
                 n_repeats: int = 100):
        self.configs = configs
        self.operators = operators
        self.n_warmup = n_warmup
        self.n_repeats = n_repeats
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for benchmarking.")
        self.device = torch.device("cuda")
        print(f"Running on device: {torch.cuda.get_device_name(self.device)}")

    def _time_operator(self, op: OperatorWrapper, config: BenchmarkConfig) -> dict:
        """对单个算子和单个配置进行计时和性能计算"""
        
        # 准备输入数据 (使用FP32模拟通用输入)
        a = torch.randn(config.batch_size, config.m, config.k, device=self.device)
        # 对于B矩阵，我们在prepare阶段已经处理，这里只做样子，实际执行时用的是打包好的
        b = torch.randn(config.batch_size, config.k, config.n, device=self.device)

        # 算子特定的准备工作
        op.prepare(config, self.device)

        # 预热
        for _ in range(self.n_warmup):
            op.execute(a, b)
        torch.cuda.synchronize()

        # 精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(self.n_repeats):
            op.execute(a, b)
        end_event.record()
        
        torch.cuda.synchronize()

        # 计算性能
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = elapsed_time_ms / self.n_repeats
        
        # 计算TOPS (Tera Operations Per Second)
        # GEMM的计算量是 2 * M * N * K
        total_ops = 2 * config.m * config.n * config.k * config.batch_size
        tops = (total_ops / (avg_latency_ms / 1000)) / 1e12

        return {
            "latency_ms": round(avg_latency_ms, 4),
            "tops": round(tops, 2)
        }

    def run(self) -> pd.DataFrame:
        """执行所有测试并返回结果"""
        results = []
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=len(self.operators) * len(self.configs), desc="Benchmarking")
        
        for op in self.operators:
            for config in self.configs:
                pbar.set_description(f"Testing {op.name()} (M={config.m}, B={config.batch_size})")
                try:
                    perf_metrics = self._time_operator(op, config)
                    
                    # 记录结果
                    result_row = {
                        "operator": op.name(),
                        "M": config.m,
                        "N": config.n,
                        "K": config.k,
                        "batch_size": config.batch_size,
                        **perf_metrics
                    }
                    results.append(result_row)
                except Exception as e:
                    print(f"Error benchmarking {op.name()} with config {config}: {e}")
                
                pbar.update(1)

        pbar.close()
        return pd.DataFrame(results)