import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from contextlib import contextmanager

# Settings
os.environ['USE_JIT_COMPILE'] = '1'
torch.set_printoptions(precision=4, sci_mode=False)
LOG_VERBOSE = False

# Prepare build dirs
THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = Path(THIS_SCRIPT_DIR, "../src").resolve()
BUILD_DIR = Path(THIS_SCRIPT_DIR, "../build").resolve()
BUILD_JIT_DIR = Path(THIS_SCRIPT_DIR, "../build-jit").resolve()
TEST_DATA_DIR = Path(THIS_SCRIPT_DIR, "./test_data").resolve()

BUILD_DIR.mkdir(exist_ok=True)
BUILD_JIT_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True)

APPLY_BCV_SOURCE = Path.joinpath(SRC_DIR, "applyBCV.cpp").resolve()
APPLY_BCV_DYLIB = Path.joinpath(BUILD_DIR, "liblibtorch-applyBCV.dylib").resolve()

# Prepare cpp modules
if os.environ.get('USE_JIT_COMPILE', None) == '1':
    # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
    # os.environ['CXX'] = "/usr/local/bin/x86_64-apple-darwin20-gcc-11.2.0"
    APPLY_BCV_MODULE = load(name="applyBCV_module", sources=[APPLY_BCV_SOURCE], build_directory=BUILD_JIT_DIR,
        # extra_cflags=["-O3", "-std=c++14", "-mavx2", "-msse4.2", "-pthread"], #"-fopenmp"],
        # extra_ldflags=[],#"-lz"],
        verbose=True)

else:
    # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
    torch.ops.load_library(APPLY_BCV_DYLIB)
    APPLY_BCV_MODULE = torch.ops.deeds_applyBCV

def test_equal_tensors(tensor_a, tensor_b):
    return torch.allclose(tensor_a, tensor_b,
        rtol=1e-05, atol=1e-08, equal_nan=False
    ), "Tensors do not match"



def log_decorator(verbose=False):
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if verbose:
                print(args, kwargs)
            print(f"Calling Function: {func.__name__} from {fn.parent}")
            result = func(*args, **kwargs)
            print(result)
            return result
        return wrapper
    return actual_decorator



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
