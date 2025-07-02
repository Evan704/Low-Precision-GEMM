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

    MODEL_SIZES = {
        "Llama-7B":  {"k": 4096, "n": 11008},
        "Llama-13B": {"k": 5120, "n": 13824},
        "Llama-70B": {"k": 8192, "n": 28672},
    }

    # 2. 定义我们想测试的LLM推理批次大小
    LLM_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    # 3. 动态生成所有测试配置
    benchmark_configs = []
    for model_name, dims in MODEL_SIZES.items():
        for batch in LLM_BATCH_SIZES:
            benchmark_configs.append(
                BenchmarkConfig(
                    m=batch,
                    k=dims["k"],
                    n=dims["n"],
                    model_name=model_name
                )
            )

    # --- 2. 实例化所有需要测试的算子 ---
    # 以后你只需要在这里添加新的算子实例即可
    operators_to_test = [
        PyTorchFP16GEMM(),
        PyTorchFP32GEMM(),
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