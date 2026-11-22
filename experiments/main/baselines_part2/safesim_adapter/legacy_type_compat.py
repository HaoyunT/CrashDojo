"""Provide annotation-only aliases for old NumPy; numerical code stays native."""
import importlib.util
import sys
import types
import typing


def install():
    if importlib.util.find_spec('numpy.typing') is not None:
        return dict(numpy_typing_annotation_shim=False)
    import numpy as np

    class NDArray:
        def __class_getitem__(cls, item):
            return np.ndarray

    module = types.ModuleType('numpy.typing')
    module.NDArray = NDArray
    module.ArrayLike = typing.Any
    module.DTypeLike = typing.Any
    sys.modules['numpy.typing'] = module
    np.typing = module
    return dict(numpy_typing_annotation_shim=True, numerical_operations_modified=False)
