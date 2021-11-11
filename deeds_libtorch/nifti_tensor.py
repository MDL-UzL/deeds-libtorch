import sys
import os
import nibabel as nib
import numpy as np
import torch 
#reading img
def read_Nifti(path,img_name,**kwargs):
    file_name=os.path.join(path,img_name)
    image=nib.load(file_name)
    header=image.shape()
    print("complete header of the image:",format(header))
    return image,header
#convert to tensor
def convert_img_tensor(nii_img,**kwargs):
    np_img=np.array(nii_img.dataobj)
    img_tensor=torch.from_numpy(np_img)
    img_tensor.unsqueeze()
    img_tensor.unsqueeze()
    tensr_size=img_tensor.size()
    print("size of the img_tensor:",format(tensr_size))
    return img_tensor,tensr_size