import nibabel as nib
import os
import sys
import numpy as np
import pandas as pd
import torch



def read_Nifti(path,img_name,**kwargs):
    #function to read a niftii file
    #returns-Tensor
    if os.path.exists(path):
        file_name=os.path.join(path,img_name)
        image=nib.load(file_name)
        np_img=np.array(image.dataobj)
        img_tensor=torch.from_numpy(np_img)
        return img_tensor
    else:
        print('Read file error-Did not find' + img_name)
    

    

def read_File(path,flow_name):
    if os.path.exists(path):
        file_name=os.path.join(path,flow_name)
        fn=open(file_name,'rb')
        flow_field=np.fromfile(fn,dtype=np.float32) #returns a 1d list- need reshaping
        return flow_field
    else:
        print('Read file error-Did not find' + flow_name)


toy_tensor=read_Nifti('/Users/sreejithkk/Internship/deeds-libtorch/','toy_nifti.nii.gz')
flow_field=read_File('/Users/sreejithkk/Internship/deeds-libtorch/','inputflow.dat')
print('flow field shape',flow_field)


    



