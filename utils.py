import torch

def quantize_to_int8(tensor: torch.Tensor):
    """
    对一个FP32张量进行对称量化，转换为INT8。
    
    返回:
        quantized_tensor (torch.Tensor): INT8类型的量化后张量。
        scale (float): 用于反量化的缩放因子。
    """
    abs_max = torch.max(torch.abs(tensor))
    
    scale = abs_max / 127.0
    
    quantized_tensor = torch.round(tensor / scale).to(torch.int8)
    
    return quantized_tensor, scale