
import os
import torch
from torch.functional import align_tensors
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def interp3d(input_img,output_size):#not sure about the output size
    if input_img.dim() == 3 :
        input_img=input_img.unsqueeze(0).unsqueeze(0)
        tri_inter3d=F.interpolate(input_img,mode='trilinear',size=output_size,align_corners=True).squeeze(0).squeeze(0)
    else:
        tri_inter3d=F.interpolate(input_img,mode='trilinear',size=output_size,align_corners=True)
    return tri_inter3d

def jacobians(x_disp_field, y_disp_field, z_disp_field):
    # Returns n=DxHxW jacobian matrices (3x3) of displacements per dimension
    #
    # Jacobian for a single data point (x,y,z) is
    #
    #   d_x_disp_field/dx  d_x_disp_field/dy  d_x_disp_field/dz
    #   d_y_disp_field/dx  d_y_disp_field/dy  d_y_disp_field/dz
    #   d_z_disp_field/dx  d_z_disp_field/dy  d_z_disp_field/dz

    assert x_disp_field.shape == y_disp_field.shape == z_disp_field.shape, \
        "Displacement field sizes must match."

    D, H, W = x_disp_field.shape

    # Prepare convolutions
    WEIGHTS = torch.tensor([[-.5, 0., .5]])
    KERNEL_SIZE = WEIGHTS.numel()

    ## Reshape weights for x,y,z dimension
    X_WEIGHTS = WEIGHTS.view(1, 1, 1, KERNEL_SIZE).repeat(1, 1, 1, 1, 1)
    Y_WEIGHTS = WEIGHTS.view(1, 1, KERNEL_SIZE, 1).repeat(1, 1, 1, 1, 1)
    Z_WEIGHTS = WEIGHTS.view(1, KERNEL_SIZE, 1, 1).repeat(1, 1, 1, 1, 1)

    def jacobians_row_entries(disp):
        # Convolute d_disp over d_x - change of displacement per x dimension
        disp_field_envelope_x = torch.cat(
            [
                disp[:,:,0:1],
                disp,
                disp[:,:,-1:]
            ],
            dim=2)
        d_disp_over_dx = torch.nn.functional.conv3d(
            disp_field_envelope_x.reshape(1,1,D,H,W+2), X_WEIGHTS
        ).reshape(D,H,W)

        # Convolute d_disp over d_y - change of displacement per y dimension
        disp_field_envelope_y = torch.cat(
            [
                disp[:,0:1,:],
                disp,
                disp[:,-1:,:]
            ],
            dim=1)
        d_disp_over_dy = torch.nn.functional.conv3d(
            disp_field_envelope_y.reshape(1,1,D,H+2,W), Y_WEIGHTS
        ).reshape(D,H,W)

        # Convolute d_disp over d_z - change of displacement per z dimension
        disp_field_envelope_z = torch.cat(
            [
                disp[0:1,:,:],
                disp,
                disp[-1:,:,:]
            ],
            dim=0)
        d_disp_over_dz = torch.nn.functional.conv3d(
            disp_field_envelope_z.reshape(1,1,D+2,H,W), Z_WEIGHTS
        ).reshape(D,H,W)

        # TODO Check why d_disp_over_dy and d_disp_over_dx need to be swapped?
        # This produces same result as deeds but I assume its a mistake.
        # J<0 count is halved when dx, dy, dz order is used
        return (d_disp_over_dy, d_disp_over_dx, d_disp_over_dz)

    J11, J12, J13  = jacobians_row_entries(x_disp_field) # First row
    J21, J22, J23  = jacobians_row_entries(y_disp_field) # Second row
    J31, J32, J33  = jacobians_row_entries(z_disp_field) # Third row

    _jacobians = torch.stack(
        [J11, J12, J13, J21, J22, J23, J31, J32, J33]
    ).reshape(3, 3, D, H, W)

    return _jacobians



def std_det_jacobians(x_disp_field, y_disp_field, z_disp_field, factor):
    # Calculate std(determinant(jacobians)) of displacement fields

    # Get jacobians
    _jacobians = jacobians(x_disp_field, y_disp_field, z_disp_field)

    # Calculate determinant and std
    jac_square_len = _jacobians.shape[0]
    spatial_len = int(_jacobians.numel()/(jac_square_len**2))

    _jacobians  = 1 / factor * _jacobians
    _jacobians[0,0]+=1
    _jacobians[1,1]+=1
    _jacobians[2,2]+=1

    jac_deteterminants = torch.det(
        _jacobians.reshape(jac_square_len,jac_square_len,spatial_len).permute(2,0,1)
    )
    std_det_jacobians = jac_deteterminants.std()

    # TODO Probably move print method elsewhere
    print(
        f"mean(J)={jac_deteterminants.mean():.6f}",
        f"std(J)={jac_deteterminants.std():.6f}",
        f"(J<0)={(jac_deteterminants<0).sum()/spatial_len*100:.5f}%")

    return std_det_jacobians



def gaussian_filter(kernel_size,sigma,dim=3):

    #Args:Kernel_size=length of gaussian filter
    #    Sigma=standard deviation
    #   dim=3-Expecting a 3D image
    #returns a gaussian filter

    #creating a 3d grid for filter
    kernel_size=[kernel_size]*dim
    sigma=[sigma]*dim
    kernel=1
    filter_3d=torch.meshgrid([torch.arange(size,dtype=torch.float32) for size in kernel_size])#a 3d grid is inititalized
    for size,sd,grid in zip(kernel_size,sigma,filter_3d):
        mu=(size-1)/2
        kernel*=torch.exp(-((grid - mu) / (2 * sd)) ** 2)#Gaussian_kernel
    kernel = kernel.clone()/ torch.sum(kernel.clone())#normalizing
    kernel=kernel.view(1,1,*kernel.size())#for depthwise convolution
    kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))
    return kernel


def vol_filter(image_in,sigma,kernel_sz=1,dim=3):
    #returns a gaussian smooth image
    weights=gaussian_filter(kernel_sz,sigma)
    if dim==1:
        conv=F.conv1d
    elif dim==2:
        conv=F.conv2d
    elif dim==3:
        conv=F.conv3d
    #-to check-input.dim=[D,H,W]or[1,1,D,H,W],for 3d conv,dim must be [1,1,D,H,W]
    image_in=image_in.unsqueeze(0).unsqueeze(0)
    image_out=conv(image_in,weights)
    return image_out


def consistentMappingCL(x_disp_field,y_disp_field,z_disp_field,x2_disp_field,y2_disp_field,z2_disp_field,factor):
    assert x_disp_field.shape == y_disp_field.shape == z_disp_field.shape, \
        "Displacement field sizes must match."
    
    assert x2_disp_field.shape == y2_disp_field.shape == z2_disp_field.shape, \
        "Displacement field sizes must match."
    D,H,W=x_disp_field.shape

    #preparing variables 
    factor_inv=1.0/factor

    #Creating Flow field for forward mapping
    disp_field_envelope_x_temp = torch.cat([x_disp_field[:,:,0:1],x_disp_field,x_disp_field[:,:,-1:]],dim=2)
    disp_field_envelope_y_temp = torch.cat([y_disp_field[:,:,0:1],y_disp_field,y_disp_field[:,:,-1:]],dim=1)
    disp_field_envelope_z_temp = torch.cat([z_disp_field[:,:,0:1],z_disp_field,z_disp_field[:,:,-1:]],dim=0)

    #multiplying with the factor
    disp_field_envelope_x_inv=torch.mul(disp_field_envelope_x_temp,factor_inv)
    disp_field_envelope_y_inv=torch.mul(disp_field_envelope_y_temp,factor_inv)
    disp_field_envelope_z_inv=torch.mul(disp_field_envelope_z_temp,factor_inv)

    #interpolating.......
    disp_field_envelope_x_inv=interp3d(disp_field_envelope_x_inv,output_size=(D,H,W))
    disp_field_envelope_y_inv=interp3d(disp_field_envelope_y_inv,output_size=(D,H,W))
    disp_field_envelope_z_inv=interp3d(disp_field_envelope_z_inv,output_size=(D,H,W))

    #some regularisation
    disp_field_envelope_x=torch.mul(disp_field_envelope_x_inv,0.5)-torch.mul(disp_field_envelope_x_temp,-0.5)
    disp_field_envelope_y=torch.mul(disp_field_envelope_y_inv,0.5)-torch.mul(disp_field_envelope_y_temp,-0.5)
    disp_field_envelope_z=torch.mul(disp_field_envelope_z_inv,0.5)-torch.mul(disp_field_envelope_z_temp,-0.5)

    #Creating 2nd Flow field for inverse mapping
    disp_field_envelope_x_temp_2 = torch.cat([x2_disp_field[:,:,0:1],x_disp_field,x_disp_field[:,:,-1:]],dim=2)
    disp_field_envelope_y_temp_2 = torch.cat([y2_disp_field[:,:,0:1],y_disp_field,y_disp_field[:,:,-1:]],dim=1)
    disp_field_envelope_z_temp_2 = torch.cat([z2_disp_field[:,:,0:1],z_disp_field,z_disp_field[:,:,-1:]],dim=0)

    #multiplying with the factor
    disp_field_envelope_x_inv_2=torch.mul(disp_field_envelope_x_temp_2,factor_inv)
    disp_field_envelope_y_inv_2=torch.mul(disp_field_envelope_y_temp_2,factor_inv)
    disp_field_envelope_z_inv_2=torch.mul(disp_field_envelope_z_temp_2,factor_inv)

    #interpolating.......
    disp_field_envelope_x_inv_2=interp3d(disp_field_envelope_x_inv_2,output_size=(D,H,W))
    disp_field_envelope_y_inv_2=interp3d(disp_field_envelope_y_inv_2,output_size=(D,H,W))
    disp_field_envelope_z_inv_2=interp3d(disp_field_envelope_z_inv_2,output_size=(D,H,W))

    #some regularisation
    disp_field_envelope_x_2=torch.mul(disp_field_envelope_x_inv_2,0.5)-torch.mul(disp_field_envelope_x_temp,-0.5)
    disp_field_envelope_y_2=torch.mul(disp_field_envelope_y_inv_2,0.5)-torch.mul(disp_field_envelope_y_temp,-0.5)
    disp_field_envelope_z_2=torch.mul(disp_field_envelope_z_inv_2,0.5)-torch.mul(disp_field_envelope_z_temp,-0.5)

    #multiplying with the factor
    disp_field_envelope_x*=factor
    disp_field_envelope_y*=factor
    disp_field_envelope_z*=factor
    disp_field_envelope_x_2*=factor
    disp_field_envelope_y_2*=factor
    disp_field_envelope_z_2*=factor

    return disp_field_envelope_x,disp_field_envelope_y,disp_field_envelope_z,disp_field_envelope_x_2,disp_field_envelope_y_2,disp_field_envelope_z_2


    

    



    

    