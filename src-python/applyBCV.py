import argparse
import torch

def main():
    torch.ops.load_library("build/libapplyBCV.dylib")
    torch.ops.deeds_applyBCV.applyBCV_main()

if __name__ == '__main__':
    main()
