import sys
import argparse
import torch

def main():
    torch.ops.load_library("build/libapplyBCV.dylib")
    torch.ops.deeds_applyBCV.applyBCV_main(len(sys.argv), sys.argv)
    # print(len(sys.argv))
    # print(sys.argv)
if __name__ == '__main__':
    main()
