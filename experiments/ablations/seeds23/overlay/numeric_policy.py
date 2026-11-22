"""Enforce inference precision before CUDA initialization and each replan."""
import os


def configure():
    required = dict(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', NVIDIA_TF32_OVERRIDE='0',
                    OPENBLAS_CORETYPE='Haswell', NPY_DISABLE_CPU_FEATURES='AVX512F',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name, value in required.items():
        if os.environ.get(name) != value:
            raise RuntimeError('Set %s=%s before starting the interpreter' % (name, value))
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return dict(environment=required, cuda_matmul_TF32=False, cudnn_TF32=False,
                cudnn_benchmark=False, deterministic_algorithms=True, warn_only=True)
