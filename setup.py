import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- (可选) 检查并设置 DISTUTILS_USE_SDK 环境变量 ---
# 这一步是为了避免 "VC environment is activated but..." 的警告。
# 也可以在命令行中手动设置: set DISTUTILS_USE_SDK=1
if sys.platform == 'win32':
    os.environ['DISTUTILS_USE_SDK'] = '1'

# --- 路径定义 ---
# 获取当前文件所在目录，并拼接 cutlass 子目录的路径
try:
    cutlass_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cutlass')
except NameError:
    # 如果在交互式环境中运行，__file__ 可能未定义
    cutlass_dir = os.path.join(os.getcwd(), 'cutlass')

# 定义需要包含的 CUTLASS 头文件目录
cutlass_include_dirs = [
    os.path.join(cutlass_dir, 'include'),
    os.path.join(cutlass_dir, 'tools/util/include')
]

# --- 编译参数定义 ---
# 根据 issue 的结论，为 MSVC 定义必需的编译标志
# '-Xcompiler' 用于将标志传递给 nvcc 调用的主机编译器 (cl.exe)
nvcc_flags = [
    # 1. 为主机编译器(MSVC)设置 C++17 标准
    '-Xcompiler', '/std:c++17',

    # 2. (关键) 确保 MSVC 正确设置 __cplusplus 宏，以便 CUTLASS 识别 C++17
    '-Xcompiler', '/Zc:__cplusplus',

    # 3. 为设备编译器(nvcc自身)设置 C++17 标准
    '-std=c++17',
    
    # 4. 设置目标 GPU 的计算能力和 SM 版本 (Turing 架构)
    #    根据您的需要可以添加更多架构, 例如:
    #    '-gencode=arch=compute_86,code=sm_86', # Ampere
    #    '-gencode=arch=compute_90,code=sm_90', # Hopper
    '-gencode=arch=compute_75,code=sm_75'
]


# --- Setuptools 入口 ---
setup(
    name='cutlass_gemm_ext',
    version='1.0',
    author='Evan704',
    description='A PyTorch C++ extension using CUTLASS',
    
    # 定义要编译的扩展模块
    ext_modules=[
        CUDAExtension(
            name='cutlass_gemm_ext',
            sources=['cutlass_gemm.cu'],
            include_dirs=cutlass_include_dirs,
            
            # 传入我们定义好的编译参数
            extra_compile_args={
                'cxx': [],  # 在 Windows 上，所有标志通过 nvcc 传递，cxx 留空
                'nvcc': nvcc_flags
            }
        )
    ],
    
    # 指定用于构建扩展的命令类
    cmdclass={
        'build_ext': BuildExtension
    }
)