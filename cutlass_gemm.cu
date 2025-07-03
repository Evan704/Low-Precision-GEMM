#include <iostream>
#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

// ##################################################################################
// #                                  FP32 SGEMM                                    #
// ##################################################################################

using Gemm_FP32_RowMajor_RowMajor = cutlass::gemm::device::Gemm<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2
>;

void sgemm_cutlass(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "Tensors must be on CUDA device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "Tensors must be contiguous");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    Gemm_FP32_RowMajor_RowMajor gemm_op;
    cutlass::Status status = gemm_op({
        problem_size,
        {(float*)a.data_ptr(), K},
        {(float*)b.data_ptr(), N},
        {(float*)c.data_ptr(), N},
        {(float*)c.data_ptr(), N},
        {1.0f, 0.0f}
    });

    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS FP32 GEMM initialization failed");
    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS FP32 GEMM execution failed");
}

// ##################################################################################
// #                                  FP16 HGEMM                                    #
// ##################################################################################

using Gemm_FP16_TensorOp = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    8, // AlignmentA
    8  // AlignmentB
>;

void hgemm_cutlass(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "Tensors must be on CUDA device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "Tensors must be contiguous");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    Gemm_FP16_TensorOp gemm_op;
    cutlass::Status status = gemm_op({
        problem_size,
        {(cutlass::half_t*)a.data_ptr(), K},
        {(cutlass::half_t*)b.data_ptr(), N},
        {(cutlass::half_t*)c.data_ptr(), N},
        {(cutlass::half_t*)c.data_ptr(), N},
        {1.0f, 0.0f}
    });

    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS FP16 GEMM initialization failed");
    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS FP16 GEMM execution failed");
}

// ##################################################################################
// #                                  INT8 IGEMM                                    #
// ##################################################################################

// Defines INT8 GEMM kernel using Tensor Cores
using Gemm_INT8_TensorOp = cutlass::gemm::device::Gemm<
    int8_t,                                 // ElementA
    cutlass::layout::RowMajor,              // LayoutA
    int8_t,                                 // ElementB
    cutlass::layout::RowMajor,              // LayoutB
    int8_t,                                 // ElementC (output)
    cutlass::layout::RowMajor,              // LayoutC
    int32_t,                                // ElementAccumulator
    cutlass::arch::OpClassTensorOp,         // OpClass
    cutlass::arch::Sm75,                    // ArchTag
    cutlass::gemm::GemmShape<128, 256, 64>, // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 64>,   // WarpShape
    cutlass::gemm::GemmShape<8, 8, 16>,     // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        int8_t,
        16, // ElementsPerAccess must match Alignment
        int32_t,
        float
    >,                                      // Epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, // ThreadblockSwizzle
    4,                                      // Stages
    16,                                     // AlignmentA
    16,                                     // AlignmentB
    false,                                  // Split-K Serial
    cutlass::arch::OpMultiplyAddSaturate    // (CRITICAL FIX) Operator
>;

void igemm_cutlass(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "Tensors must be on CUDA device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "Tensors must be contiguous");

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    Gemm_INT8_TensorOp gemm_op;
    float alpha = 1.0f;
    float beta = 0.0f;

    cutlass::Status status = gemm_op({
        problem_size,
        {(int8_t*)a.data_ptr(), K},
        {(int8_t*)b.data_ptr(), N},
        {(int8_t*)c.data_ptr(), N},
        {(int8_t*)c.data_ptr(), N},
        {alpha, beta}
    });

    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS INT8 GEMM initialization failed");
    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS INT8 GEMM execution failed");
}

// ##################################################################################
// #                                PYBIND11 BINDINGS                               #
// ##################################################################################

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm", &sgemm_cutlass, "CUTLASS FP32 GEMM");
    m.def("hgemm", &hgemm_cutlass, "CUTLASS FP16 GEMM");
    m.def("igemm", &igemm_cutlass, "CUTLASS INT8 GEMM");
}