import nibabel as nib
import os
import sys
from pathlib import Path
import numpy as np
import torch



def read_nifti(_path):
    #function to read a niftii file
    #returns-Tensor
    if os.path.exists(_path):
        image=nib.load(_path)
        header = image.header
        affine = image.affine
        data_tensor = torch.from_numpy(image.get_fdata())
        return data_tensor, header, affine
    else:
        raise FileNotFoundError(f"Cannot find nifti file '{_path}'")

def save_nifti(data_tensor, _path, header=None, affine=None, nifti_version=None):
    _path = Path(_path)
    if header == None:
        assert nifti_version in ['1.0', '2.0']
    else:
        assert nifti_version in ['1.0', '2.0', None]
        if nifti_version == None:
            if isinstance(header, nib.nifti1.Nifti1Header):
                nifty_type = nib.nifti1.Nifti1Image
            elif isinstance(header, nib.nifti2.Nifti2Header):
                nifty_type = nib.nifti2.Nifti2Image
        elif nifti_version == '1.0':
             nifty_type = nib.nifti1.Nifti1Image
        elif nifti_version == '2.0':
            nifty_type = nib.nifti2.Nifti2Image()


    if header is None and affine is None:
        affine = np.eye(4,4)
    elif affine is None:
        affine = header.get_best_affine()

    nifti_img = nifty_type(data_tensor, affine, header)
    # Override data type based on data_tensor dtype
    nifti_img.set_data_dtype(data_tensor.numpy().dtype)

    if _path.parent.exists():
        nib.save(nifti_img, _path)
    else:
        raise IOError(f"Base directory '{_path.parent}' does not exist. Saving nifti failed.")



def read_affine_file(_path):
    if os.path.exists(_path):
        affine_mat = np.fromfile(_path, dtype=np.float32, sep=" ") #returns a 1d list- need reshaping
        affine_mat = torch.from_numpy(affine_mat).reshape(4,4)
        return affine_mat
    else:
        raise FileNotFoundError(f"Cannot find affine matrix file '{_path}'")

    return None