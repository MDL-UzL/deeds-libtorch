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

CPP_APPLY_BCV_SOURCE = Path.joinpath(SRC_DIR, "applyBCV.cpp").resolve()
DEEDS_SOURCE = Path.joinpath(SRC_DIR, "deedsBCV0.cpp").resolve()

CPP_APPLY_BCV_DYLIB = Path.joinpath(BUILD_DIR, "liblib-applyBCV.so").resolve()
DEEDS_DYLIB = Path.joinpath(BUILD_DIR, "liblib-deedsBCV.so").resolve()

# Prepare cpp modules
if os.environ.get('USE_JIT_COMPILE', None) == '1':
    # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
    # os.environ['CXX'] = "/usr/local/bin/x86_64-apple-darwin20-gcc-11.2.0"

    extra_cflags = [
        # "-O3",
        "-std=c++14",
        # "-mavx2",
        # "-msse4.2",
        # "-pthread",
        "-g",
        # "-fopenmp"
    ]

    extra_ldflags = [
        # "-L/usr/lib",
        "-lz"
        ]

    CPP_APPLY_BCV_MODULE = load(name="cpp_apply_bcv_module", sources=[CPP_APPLY_BCV_SOURCE], build_directory=BUILD_JIT_DIR,
        extra_cflags=extra_cflags, extra_ldflags=extra_ldflags,
        verbose=True)
    CPP_DEEDS_MODULE = load(name="cpp_deeds_module", sources=[DEEDS_SOURCE], build_directory=BUILD_JIT_DIR,
        extra_cflags=extra_cflags, extra_ldflags=extra_ldflags,
        verbose=True)

else:
    # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
    torch.ops.load_library(CPP_APPLY_BCV_DYLIB)
    CPP_APPLY_BCV_MODULE = torch.ops.cpp_applyBCV

    torch.ops.load_library(DEEDS_DYLIB)
    CPP_DEEDS_MODULE = torch.ops.cpp_deeds



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


# MIND specific helpers

def mind_extract_features(binaries):
    BITS_PER_FEATURE = 5
    NUM_FEAURES = 12

    binaries = binaries[...,:BITS_PER_FEATURE*NUM_FEAURES]
    D,H,W,DIGITS = binaries.shape

    sums = torch.empty((D,H,W,12))
    for feature_idx, d_start_idx in enumerate(range(0,DIGITS,5)):
        sums[...,feature_idx] = binaries[..., d_start_idx:d_start_idx+5].sum(dim=-1)
    return sums

def mind_compress_features(feat_tensor):
    bit_str = ""
    for feature_idx, feat_val in enumerate(feat_tensor):
        set_bits = ('1' * feat_val.item()).ljust(5, "0")
        bit_str += set_bits
    bit_str = bit_str.rjust(64, "0")
    return [int(b) for b in bit_str]

def unpackbits(x, bits):
    # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    x = x.long()
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def packbits(data, dtype):
    if dtype == torch.long:
        stype = "L"
    data = data.clone()
    BIT_DIM = data.shape[-1]
    SPATIAL_SHAPE = data.shape[0:-1]
    out = torch.empty(SPATIAL_SHAPE).view(-1).to(dtype)

    for t_idx, bit_entry in enumerate(data.view(-1, BIT_DIM)):
        bita = bitarray("".join([str(b) for b in bit_entry.tolist()]), endian='little')
        bit_entry = struct.unpack(stype, bita)[0]
        out[t_idx] = bit_entry
    return out.reshape(SPATIAL_SHAPE)

def mind_twelve_to_long(mind_tensor):
    MIND_DIM = 12
    assert mind_tensor.shape[-1] == MIND_DIM
    mind_tensor = mind_tensor.clone().long()
    SPATIAL_SHAPE = mind_tensor.shape[0:-1]
    out = torch.empty(SPATIAL_SHAPE).view(-1).long()

    for t_idx, mind_twelve in enumerate(mind_tensor.reshape(-1, MIND_DIM)):
        bits = torch.tensor(compress_features(mind_twelve))
        long_value = packbits(bits, torch.long)
        out[t_idx] = long_value
    return out.reshape(SPATIAL_SHAPE)

def mind_long_to_twelve(mind_tensor):
    assert mind_tensor.dim() == 3
    mind_tensor = mind_tensor.clone()
    SPATIAL_SHAPE = mind_tensor.shape[0:-1]
    out = torch.empty((12,)+SPATIAL_SHAPE).view(-1).float()

    mind_bits = unpackbits(mind_tensor, 64)
    mind_twelve = mind_extract_features(mind_bits).permute(3,0,1,2)

    return mind_twelve