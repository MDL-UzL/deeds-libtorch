import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from contextlib import contextmanager

# Settings
os.environ['USE_JIT_COMPILE'] = '1'
torch.set_printoptions(precision=4, sci_mode=False)
LOG_VERBOSE = True

# Prepare build dirs
THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = Path(THIS_SCRIPT_DIR, "../src").resolve()
BUILD_DIR = Path(THIS_SCRIPT_DIR, "../build").resolve()
BUILD_JIT_DIR = Path(THIS_SCRIPT_DIR, "../build-jit").resolve()
TEST_DATA_DIR = Path(THIS_SCRIPT_DIR, "./test_data").resolve()

BUILD_DIR.mkdir(exist_ok=True)
BUILD_JIT_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True)

CPP_APPLY_BCV_SOURCE = Path.joinpath(SRC_DIR, "applyBCV.cpp").resolve()
CPP_APPLY_BCV_DYLIB = Path.joinpath(BUILD_DIR, "liblibtorch-applyBCV.dylib").resolve()

CPP_TRANSFORMATIONS_SOURCE = Path.joinpath(SRC_DIR, "transformations.h").resolve()
CPP_TRANSFORMATIONS_DYLIB = Path.joinpath(BUILD_DIR, "liblibtorch-transformations.dylib").resolve()

# Prepare cpp modules
if os.environ.get('USE_JIT_COMPILE', None) == '1':
    extra_cflags = ["-O3", "-std=c++14", "-mavx2", "-msse4.2", "-pthread"] #"-fopenmp"]
    extra_ldflags = [], #"-lz"]

    # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
    # os.environ['CXX'] = "/usr/local/bin/x86_64-apple-darwin20-gcc-11.2.0"
    CPP_APPLY_BCV_MODULE = load(name="cpp_apply_bcv_module", sources=[CPP_APPLY_BCV_SOURCE], build_directory=BUILD_JIT_DIR,
        # extra_cflags=extra_cflags, extra_ldflags=extra_ld_flags,
        verbose=True)

else:
    # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
    torch.ops.load_library(CPP_APPLY_BCV_DYLIB)
    CPP_APPLY_BCV_MODULE = torch.ops.cpp_applyBCV



def test_equal_tensors(tensor_a, tensor_b, lazy=False):
    RTOL = 1e-03
    ATOL = 1e-06
    UNEQ_MAX_RATIO = 0.005

    if lazy:
        res = torch.isclose(tensor_a, tensor_b,
        rtol=RTOL, atol=ATOL, equal_nan=False
    )
        uniques, cnt = res.unique(return_counts=True)
        if uniques.numel() == 2:
            (uneq, eq) = cnt
            return uneq/eq < UNEQ_MAX_RATIO
        return all(uniques)

    return torch.allclose(tensor_a, tensor_b,
        rtol=RTOL, atol=ATOL, equal_nan=False
    )



def log_wrapper(func, *args, **kwargs):
    print(f"Calling Function: {func.__name__}")
    if LOG_VERBOSE:
        print("args are:")
        for rg in args:
            print(rg)
        print()
        print(print("kwargs are:"))
        for kwrg in kwargs.items():
            print(kwrg)
        print()

    result = func(*args, **kwargs)

    if LOG_VERBOSE:
        print("Output is:")
        print(result)
        print()

    return result
