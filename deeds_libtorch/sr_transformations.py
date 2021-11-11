import os 
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#input_img description-tensor[m,n,o]

def interp3d(input_img,*args,**kwargs):
    tri_inter3d=F.interpolate(input_img,mode='trilinear',scale_factor=(1,1,2,2,2))
    return tri_inter3d

def filter(*args,**kwargs):
    pass
def vol_filter(*args,**kwargs):
    pass
def jacob_energy_mtx(*args,**kwargs):
    pass
