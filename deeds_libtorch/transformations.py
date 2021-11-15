import math
import os
import torch
from torch.functional import align_tensors
import torch.nn.functional as F
from torch.utils.cpp_extension import load



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




def interp3_naive(_input, x1, y1, z1, output_size, flag):
    insz_y, insz_x, insz_z = _input.shape
    osz_y, osz_x, osz_z  = output_size

    def clamp_xyz_idx(idx_x, idx_y, idx_z):
        x_clamp = min(max(idx_x,0),insz_x-1)
        y_clamp = min(max(idx_y,0),insz_y-1)
        z_clamp = min(max(idx_z,0),insz_z-1)
        x_clamp, y_clamp, z_clamp = y_clamp, x_clamp, z_clamp
        return (x_clamp, y_clamp, z_clamp)



    interp = torch.zeros(output_size)

    for k in range(osz_z): # iterate output z
        for j in range(osz_y): # iterate output y
            for i in range(osz_x): # iterate output x
                x = int(math.floor(x1[j,i,k]))
                y = int(math.floor(y1[j,i,k]))
                z = int(math.floor(z1[j,i,k]))

                dx=float(x1[j,i,k]-x)
                dy=float(y1[j,i,k]-y)
                dz=float(z1[j,i,k]-z) # dx,dy,dz in gridded flow field relative coordinates

                if flag:
                    x+=j; y+=i; z+=k
                # Y,X,Z
                interp[j,i,k]=\
                (1.0-dx)*(1.0-dy)*(1.0-dz)*	_input[	clamp_xyz_idx(x, y, z)      ]  \
                +dx*(1.0-dy)*(1.0-dz)*		_input[	clamp_xyz_idx(x+1, y, z)	]  \
                +(1.0-dx)*dy*(1.0-dz)*		_input[	clamp_xyz_idx(x, y+1, z)    ]  \
                +(1.0-dx)*(1.0-dy)*dz*		_input[	clamp_xyz_idx(x, y, z+1)	]  \
                +(1.0-dx)*dy*dz*			_input[	clamp_xyz_idx(x, y+1, z+1)  ]  \
                +dx*(1.0-dy)*dz*			_input[	clamp_xyz_idx(x+1, y, z+1)	]  \
                +dx*dy*(1.0-dz)*			_input[	clamp_xyz_idx(x+1, y+1, z)  ]  \
                +dx*dy*dz*					_input[ clamp_xyz_idx(x+1, y+1, z+1)]



def interp3_most_naive(
			 input,
			 x1, y1, z1,
            output_shape,
			  flag):

    m2,n2,o2 = input.shape

    m,n,o =  output_shape
    interp = torch.zeros(output_shape)

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    z1 = z1.reshape(-1)
    input = input.reshape(-1)
    interp = interp.reshape(-1)

    for k in range(o):
        for j in range(n):
            for i in range(m):
                x=int(math.floor(x1[i+j*m+k*m*n]))
                y=int(math.floor(y1[i+j*m+k*m*n]))
                z=int(math.floor(z1[i+j*m+k*m*n]))
                dx=x1[i+j*m+k*m*n]-x
                dy=y1[i+j*m+k*m*n]-y
                dz=z1[i+j*m+k*m*n]-z

                if(flag):
                    x+=j; y+=i; z+=k

                interp[i+j*m+k*m*n]=\
                (1.0-dx)*(1.0-dy)*(1.0-dz)*	input[	min(max(y,0),m2-1)			+min(max(x,0),n2-1)*m2						+min(max(z,0),o2-1)*m2*n2]\
                +dx*(1.0-dy)*(1.0-dz)*		input[	min(max(y,0),m2-1)			+min(max(x+1,0),n2-1)*m2					+min(max(z,0),o2-1)*m2*n2]\
                +(1.0-dx)*dy*(1.0-dz)*		input[	min(max(y+1,0),m2-1)		+min(max(x,0),n2-1)*m2						+min(max(z,0),o2-1)*m2*n2]\
                +(1.0-dx)*(1.0-dy)*dz*		input[	min(max(y,0),m2-1)			+min(max(x,0),n2-1)*m2						+min(max(z+1,0),o2-1)*m2*n2]\
                +(1.0-dx)*dy*dz*			input[	min(max(y+1,0),m2-1)		+min(max(x,0),n2-1)*m2						+min(max(z+1,0),o2-1)*m2*n2]\
                +dx*(1.0-dy)*dz*			input[	min(max(y,0),m2-1)			+min(max(x+1,0),n2-1)*m2					+min(max(z+1,0),o2-1)*m2*n2]\
                +dx*dy*(1.0-dz)*			input[	min(max(y+1,0),m2-1)		+min(max(x+1,0),n2-1)*m2					+min(max(z,0),o2-1)*m2*n2]\
                +dx*dy*dz*					input[  min(max(y+1,0),m2-1)		+min(max(x+1,0),n2-1)*m2					+min(max(z+1,0),o2-1)*m2*n2]

    return interp.reshape(output_shape)


def consistentMappingCL(u1,v1,w1,u2,v2,w2,factor):
    #u1,v1,w1- deformation field1
    #u2,v2,w2- deformation field2
    output_shape=u1.shape
    factor_inv=1.0/factor
    u1_temp=torch.mul(u1,factor_inv)
    v1_temp=torch.mul(v1,factor_inv)
    w1_temp=torch.mul(w1,factor_inv)
    u2_temp=torch.mul(u2,factor_inv)
    v2_temp=torch.mul(v2,factor_inv)
    w2_temp=torch.mul(w2,factor_inv)

    #interpolatioing field 2 by compositing with field 1..
    u1=interp3_most_naive(u2_temp,u1_temp,v1_temp,w1_temp,output_shape,True)
    v1=interp3_most_naive(v2_temp,u1_temp,v1_temp,w1_temp,output_shape,True)
    w1=interp3_most_naive(w2_temp,u1_temp,v1_temp,w1_temp,output_shape,True)

    #composition
    u1=torch.mul(u1_temp,0.5)+torch.mul(u1,-0.5)
    v1=torch.mul(v1_temp,0.5)+torch.mul(v1,-0.5)
    w1=torch.mul(w1_temp,0.5)+torch.mul(w1,-0.5)

    #interpolating field 1 by composition with field2
    u2=interp3_most_naive(u1_temp,u2_temp,v2_temp,w2_temp,output_shape,True)
    v2=interp3_most_naive(v1_temp,u2_temp,v2_temp,w2_temp,output_shape,True)
    w2=interp3_most_naive(w1_temp,u2_temp,v2_temp,w2_temp,output_shape,True)

    #composition
    u2=torch.mul(u2_temp,0.5)+torch.mul(u2,-0.5)
    v2=torch.mul(v2_temp,0.5)+torch.mul(v2,-0.5)
    w2=torch.mul(w2_temp,0.5)+torch.mul(w2,-0.5)

    #refactoring
    u1=torch.mul(u1,factor)
    v1=torch.mul(v1,factor)
    w1=torch.mul(w1,factor)
    u2=torch.mul(u2,factor)
    v2=torch.mul(v2,factor)
    w2=torch.mul(w2,factor)

    return u1


def upsampleDeformationsCL(u1,v1,w1,u,v,w):
    #u1,v1,w1-flow field
    #u,v,w-gridded flow field
    D1,H1,W1=u1.shape
    D2,H2,W2=u.shape
    i=D1/D2
    j=H1/H2
    k=





