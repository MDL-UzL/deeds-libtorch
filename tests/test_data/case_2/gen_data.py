import nibabel
import numpy as np

SIZE_D,SIZE_H,SIZE_W = 12,24,36
nii_data = np.random.random_sample((SIZE_D,SIZE_H,SIZE_W))

affine = np.eye(4,4)
nii_img = nibabel.Nifti1Image(nii_data, affine)
nii_label = nibabel.Nifti1Image((nii_data*10).astype('short'), affine)
nibabel.save(nii_img, "toy_nifti_random_DHW_12x24x36.nii.gz")
nibabel.save(nii_label, "label_toy_nifti_random_DHW_12x24x36.nii.gz")

float_array = np.zeros((SIZE_D//4,SIZE_H//4,SIZE_W//4,3)).astype('float32')
output_file = open('DHW_12x24x36_flow_dim_3x6x9_zero_displacements.dat', 'wb')

float_array.tofile(output_file)
output_file.close()