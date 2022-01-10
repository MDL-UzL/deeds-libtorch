import nibabel as nib
import os
import sys
import numpy as np
import torch



def read_Nifti(path):
    #function to read a niftii file
    #returns-Tensor
    if os.path.exists(path):
        image=nib.load(path)
        np_img=np.array(image.dataobj)
        img_tensor=torch.from_numpy(np_img)
        return img_tensor
    else:
        print('Read file error-Did not find' + path)
    

    

def read_File(path):
    if os.path.exists(path):
        fn=open(path,'rb')
        flow_field=np.fromfile(fn,dtype=np.float32) #returns a 1d list- need reshaping
        return flow_field
    else:
        print('Read file error-Did not find' + path)





    



