import nibabel as nib
import os
import sys
import numpy as np
import torch



def read_nifti(_path):
    #function to read a niftii file
    #returns-Tensor
    if os.path.exists(_path):
        image=nib.load(_path)
        np_img=image.get_fdata()
        img_tensor=torch.from_numpy(np_img)
        return img_tensor
    else:
        print('Read file error-Did not find' + path)



def read_affine_file(_path):
    if os.path.exists(_path):
        affine_mat = np.fromfile(_path, dtype=np.float32, sep=" ") #returns a 1d list- need reshaping
        affine_mat = torch.from_numpy(affine_mat).reshape(4,4)
        return affine_mat
    else:
        raise FileNotFoundError(f"Cannot find affine matrix file '{_path}'")

    return None