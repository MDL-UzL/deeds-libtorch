import nibabel
import numpy as np

SIZE_D,SIZE_H,SIZE_W = 10,20,30
nii_data = np.random.random_sample((SIZE_D,SIZE_H,SIZE_W))

affine = np.eye(4,4)
nii_img = nibabel.Nifti1Image(nii_data, affine)
nibabel.save(nii_img, "toy_nifti_random_DHW_10x20x30.nii.gz")

float_array = np.zeros((SIZE_D//4,SIZE_H//4,SIZE_W//4,3)).astype('float32')
output_file = open('DHW_12x24x36_flow_dim_3x6x9_zero_displacements.dat', 'wb')

float_array.tofile(output_file)
output_file.close()