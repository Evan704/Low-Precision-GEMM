# run_benchmark.py

import argparse
import pandas as pd
from benchmark import BenchmarkRunner, BenchmarkConfig
from operators.fp16_op import PyTorchFP16GEMM
from operators.fp32_op import PyTorchFP32GEMM
# 当你有了新的算子实现后，在这里导入它
# from operators.my_cutlass_op import MyCutlassOp 

def main():
    parser = argparse.ArgumentParser(description="GPU GEMM Operator Benchmark Framework")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv",
                        help="Path to save the benchmark results CSV file.")
    args = parser.parse_args()

    # --- 1. 定义你的测试配置 ---
    # 你可以定义任意数量的场景
    benchmark_configs = [
        # LLM FFN-like (大M, 小N)
        BenchmarkConfig(m=4096, n=1024, k=4096, batch_size=1),
        BenchmarkConfig(m=4096, n=1024, k=4096, batch_size=8),
        BenchmarkConfig(m=4096, n=1024, k=4096, batch_size=32),
        
        # Square-like
        BenchmarkConfig(m=2048, n=2048, k=2048, batch_size=1),
        BenchmarkConfig(m=2048, n=2048, k=2048, batch_size=16),
        
        # Low-rank-like
        BenchmarkConfig(m=4096, n=4096, k=128, batch_size=1),
        BenchmarkConfig(m=4096, n=4096, k=128, batch_size=32),

        # BenchmarkConfig(m=128, n=128, k=128, batch_size=1)
    ]

    # --- 2. 实例化所有需要测试的算子 ---
    # 以后你只需要在这里添加新的算子实例即可
    operators_to_test = [
        PyTorchFP16GEMM(),
        PyTorchFP32GEMM()
        # MyCutlassOp(),  # 像这样添加你的新算子
        # MyTritonOp(),
    ]

    # --- 3. 运行测试框架 ---
    runner = BenchmarkRunner(configs=benchmark_configs, operators=operators_to_test)
    results_df = runner.run()

    # --- 4. 展示和保存结果 ---
    print("\n--- Benchmark Results ---")
    print(results_df.to_string())

    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main()