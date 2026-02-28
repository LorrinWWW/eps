import torch

from . import ops as ops

from .silu import silu as silu

device = torch.cuda.current_device()
capability = torch.cuda.get_device_capability(device)
arch_name = torch.cuda.get_device_name(device)

# Hopper 架构的计算能力为 9.0（sm_90 或 sm_90a）
IS_HOPPER = (capability == (9, 0))
IS_BLACKWELL = (capability[0] == 10)
if IS_HOPPER or IS_BLACKWELL:
    from .aok_grouped_gemm import AokGroupedGemm as AokGroupedGemm
