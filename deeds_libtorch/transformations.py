import math
import os
import torch
from torch.functional import align_tensors
import torch.nn.functional as F
from torch.utils.cpp_extension import load



def jacobians(x_disp_field, y_disp_field, z_disp_field):
    # Returns outsz_y=DxHxW jacobian matrices (3x3) of displacements per dimension
    #
    # Jacobian for a single data point (x,y,z) is
    #
    #   d_x_disp_field/dx  d_x_disp_field/dy  d_x_disp_field/dz
    #   d_y_disp_field/dx  d_y_disp_field/dy  d_y_disp_field/dz
    #   d_z_disp_field/dx  d_z_disp_field/dy  d_z_disp_field/dz

    USE_CONSISTENT_TORCH = False

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

        if not USE_CONSISTENT_TORCH:
            # TODO Check why d_disp_over_dy and d_disp_over_dx need to be swapped?
            # This produces same result as deeds but I assume its a mistake.
            # J<0 count is halved when dx, dy, dz order is used
            return (d_disp_over_dy, d_disp_over_dx, d_disp_over_dz)

        return (d_disp_over_dx, d_disp_over_dy, d_disp_over_dz)


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





def interp3(input, x1, y1, z1, output_shape, flag):

    USE_CONSISTENT_TORCH = False
    insz_x, insz_y, insz_z = input.shape
    outsz_x, outsz_y, outsz_z =  output_shape
    interp = torch.zeros(output_shape)

    if not USE_CONSISTENT_TORCH:
        x1 = x1.reshape(outsz_z, outsz_y, outsz_x).permute(2,1,0)
        y1 = y1.reshape(outsz_z, outsz_y, outsz_x).permute(2,1,0)
        z1 = z1.reshape(outsz_z, outsz_y, outsz_x).permute(2,1,0)
        input = input.reshape(insz_z,insz_y,insz_x).permute(1,2,0)

    def clamp_xyz(x,y,z):
        if USE_CONSISTENT_TORCH:
            return (
                min(max(x,0), insz_x-1),
                min(max(y,0), insz_y-1),
                min(max(z,0), insz_z-1)
            )

        return (
            # We have switched index clamping in original implementation
            min(max(x,0),insz_y-1),
            min(max(y,0),insz_x-1),
            min(max(z,0),insz_z-1)
        )

    for k in range(outsz_z):
        for j in range(outsz_y):
            for i in range(outsz_x):
                x=int(math.floor(x1[i,j,k]))
                y=int(math.floor(y1[i,j,k]))
                z=int(math.floor(z1[i,j,k]))
                dx=x1[i,j,k]-x
                dy=y1[i,j,k]-y
                dz=z1[i,j,k]-z

                if(flag):
                    if USE_CONSISTENT_TORCH:
                        x+=i
                        y+=j
                        z+=k

                    else:
                        x+=j
                        y+=i
                        z+=k

                interp[i,j,k]=\
                (1.0-dx)*(1.0-dy)*(1.0-dz)*	input[clamp_xyz(x, y, z)]\
                +dx*(1.0-dy)*(1.0-dz)*		input[clamp_xyz(x+1, y, z)]\
                +(1.0-dx)*dy*(1.0-dz)*		input[clamp_xyz(x, y+1, z)]\
                +(1.0-dx)*(1.0-dy)*dz*		input[clamp_xyz(x, y, z+1)]\
                +(1.0-dx)*dy*dz*			input[clamp_xyz(x, y+1, z+1)]\
                +dx*(1.0-dy)*dz*			input[clamp_xyz(x+1, y, z+1)]\
                +dx*dy*(1.0-dz)*			input[clamp_xyz(x+1, y+1, z)]\
                +dx*dy*dz*					input[clamp_xyz(x+1, y+1, z+1)]

    if not USE_CONSISTENT_TORCH:
        return interp.permute(2,1,0).reshape(output_shape)

    return interp



def consistentMappingCL(u1,v1,w1,u2,v2,w2,factor):
    #u1,v1,w1- deformation field1
    #u2,v2,w2- deformation field2
    output_shape=u1.shape
    factor_inv=1.0/factor
    epochs=10
    u1_temp=torch.mul(u1,factor_inv)
    v1_temp=torch.mul(v1,factor_inv)
    w1_temp=torch.mul(w1,factor_inv)
    u2_temp=torch.mul(u2,factor_inv)
    v2_temp=torch.mul(v2,factor_inv)
    w2_temp=torch.mul(w2,factor_inv)
    #iteration required

    for epoch in range(epochs):
        #interpolatioing field 2 by compositing with field 1..
        u1=interp3(u2_temp,u1_temp,v1_temp,w1_temp,output_shape,True)
        v1=interp3(v2_temp,u1_temp,v1_temp,w1_temp,output_shape,True)
        w1=interp3(w2_temp,u1_temp,v1_temp,w1_temp,output_shape,True)

        #composition
        u1=torch.mul(u1_temp,0.5)+torch.mul(u1,-0.5)
        v1=torch.mul(v1_temp,0.5)+torch.mul(v1,-0.5)
        w1=torch.mul(w1_temp,0.5)+torch.mul(w1,-0.5)

        #interpolating field 1 by composition with field2
        u2=interp3(u1_temp,u2_temp,v2_temp,w2_temp,output_shape,True)
        v2=interp3(v1_temp,u2_temp,v2_temp,w2_temp,output_shape,True)
        w2=interp3(w1_temp,u2_temp,v2_temp,w2_temp,output_shape,True)

        #composition
        u2=torch.mul(u2_temp,0.5)+torch.mul(u2,-0.5)
        v2=torch.mul(v2_temp,0.5)+torch.mul(v2,-0.5)
        w2=torch.mul(w2_temp,0.5)+torch.mul(w2,-0.5)

        #updating temporary variables

        u1_temp=u1
        v1_temp=v1
        w1_temp=w1
        u2_temp=u2
        v2_temp=v2
        w2_temp=w2



    #refactoring
    u1=torch.mul(u1,factor)
    v1=torch.mul(v1,factor)
    w1=torch.mul(w1,factor)
    u2=torch.mul(u2,factor)
    v2=torch.mul(v2,factor)
    w2=torch.mul(w2,factor)

    return u1, v1, w1, u2, v2, w2


def upsampleDeformationsCL(u1,v1,w1,u,v,w,
    output_size=None):

    USE_CONSISTENT_TORCH=False

    assert u.dim() == v.dim() == w.dim() == 3,\
        "Input displacements must be 3-dimensional."
    assert u.shape == v.shape == w.shape,\
        "Displacement field sizes must match."

    if USE_CONSISTENT_TORCH:
        u = u.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)
        w = w.unsqueeze(0).unsqueeze(0)
        return (
            torch.nn.functional.interpolate(u, size=output_size, mode='trilinear', align_corners=False),
            torch.nn.functional.interpolate(v, size=output_size, mode='trilinear', align_corners=False),
            torch.nn.functional.interpolate(w, size=output_size, mode='trilinear', align_corners=False)
        )

    #u1,v1,w1-flow field
    #u,v,w-gridded flow field
    u2=u
    v2=v
    w2=w
    D1,H1,W1=u1.shape #Full size flow field shape
    D2,H2,W2=u.shape  #gridded flow field shape

    #scaling
    scale_d=D1/D2
    scale_h=H1/H2
    scale_w=W1/W2

    #initializing helper variables
    X1=torch.zeros((D1,H1,W1))
    Y1=torch.zeros((D1,H1,W1))
    Z1=torch.zeros((D1,H1,W1))
    for k in range(D1):
        for j in range(H1):
            for i in range(W1):
                X1[i,j,k]=j/scale_h
                Y1[i,j,k]=k/scale_w
                Z1[i,j,k]=i/scale_d


    #interpolating
    u1=interp3(u2,X1,Y1,Z1,(D1,H1,W1),flag=False)
    v1=interp3(v2,X1,Y1,Z1,(D1,H1,W1),flag=False)
    w1=interp3(w2,X1,Y1,Z1,(D1,H1,W1),flag=False)

    return u1,v1,w1