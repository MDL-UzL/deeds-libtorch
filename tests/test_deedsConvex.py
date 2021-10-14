import os
import unittest
import subprocess
import shlex
import nibabel as nib
import numpy as np

DO_INIT = False
W,H,D = 140, 140, 80
AFFINE_MAT_PATH = "res/affine_matrix.txt"

class TestDeedsConvex(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
        os.chdir(THIS_SCRIPT_DIR)
        print(f"Working dir is: {os.getcwd()}")

        if DO_INIT:
            # Prepare affine mat
            subprocess.call(
                    shlex.split("./bin/linearBCV -F fixed_img.nii.gz -M moving_img.nii.gz -O out/affine")
            )

            # Prepare dummy flow
            with open('res/no_flow_displacements.dat', 'wb') as output_file:
                float_array = np.zeros((D//4,H//4,W//4,3)).astype('float32')
                float_array.tofile(output_file)

    def test1_no_affine_m_vs_m_at_lcc_img(self):
        subprocess.call(
            shlex.split("./bin/wbirLCC -F moving_img.nii -M moving_img.nii -O out/t1_ref -S moving_seg.nii")
        )
        subprocess.call(
            shlex.split("./bin/deedsConvexLCC -F moving_img.nii.gz -M moving_img.nii.gz -O out/t1_test -S moving_seg.nii.gz")
        )

        ref_img = nib.load("out/t1_ref_deformed.nii").dataobj
        test_img = nib.load("out/t1_test_deformed.nii.gz").dataobj
        np.testing.assert_allclose(ref_img, test_img, atol=1e-10)

    def test1_no_affine_m_vs_m_at_lcc_seg(self):

        ref_seg = nib.load("out/t1_ref_segment.nii").dataobj
        test_seg = nib.load("out/t1_test_deformed_seg.nii.gz").dataobj
        np.testing.assert_allclose(ref_seg, test_seg, atol=1e-10)


    def test2_no_affine_m_vs_m_at_ssc_img(self):
        subprocess.call(
            shlex.split("./bin/wbirSSC -F moving_img.nii -M moving_img.nii -O out/t2_ref -S moving_seg.nii")
        )
        subprocess.call(
            shlex.split("./bin/deedsConvexSSC -F moving_img.nii.gz -M moving_img.nii.gz -O out/t2_test -S moving_seg.nii.gz")
        )

        ref_img = nib.load("out/t2_ref_deformed.nii").dataobj
        test_img = nib.load("out/t2_test_deformed.nii.gz").dataobj
        np.testing.assert_allclose(ref_img, test_img, atol=1e-10)

    def test2_no_affine_m_vs_m_at_ssc_seg(self):
        ref_seg = nib.load("out/t2_ref_segment.nii").dataobj
        test_seg = nib.load("out/t2_test_deformed_seg.nii.gz").dataobj
        np.testing.assert_allclose(ref_seg, test_seg, atol=1e-10)


    def test3_affine_only_lcc_img(self):
        subprocess.call(
            shlex.split("./bin/applyBCVfloat -A res/affine_matrix.txt -M moving_img.nii.gz -O res/no_flow -D out/t3_refm.nii.gz")
        )
        subprocess.call(
            shlex.split("./bin/applyBCV -A res/affine_matrix.txt -M moving_seg.nii.gz -O res/no_flow -D out/t3_refs.nii.gz")
        )

        subprocess.call(
            shlex.split("./bin/deedsConvexLCC -A res/affine_matrix.txt -F out/t3_refm.nii.gz -M moving_img.nii.gz -O out/t3_test -S moving_seg.nii.gz")
        )

        ref_img = nib.load("out/t3_refm.nii.gz").dataobj
        test_img = nib.load("out/t3_test_deformed.nii.gz").dataobj
        np.testing.assert_allclose(ref_img, test_img, atol=1e-10)

    def test3_affine_only_lcc_seg(self):
        ref_seg = nib.load("out/t3_refs.nii.gz").dataobj
        test_seg = nib.load("out/t3_test_deformed_seg.nii.gz").dataobj
        np.testing.assert_allclose(ref_seg, test_seg, atol=1e-10)


    def test4_affine_only_ssc_img(self):
        subprocess.call(
            shlex.split("./bin/applyBCVfloat -A res/affine_matrix.txt -M moving_img.nii.gz -O res/no_flow -D out/t4_refm.nii.gz")
        )
        subprocess.call(
            shlex.split("./bin/applyBCV -A res/affine_matrix.txt -M moving_seg.nii.gz -O res/no_flow -D out/t4_refs.nii.gz")
        )

        subprocess.call(
            # Image is not deformed correctly
            # Seg is working properly here
            shlex.split("./bin/deedsConvexSSC -A res/affine_matrix.txt -F out/t4_refm.nii.gz -M moving_img.nii.gz -O out/t4_test -S moving_seg.nii.gz")
        )

        ref_img = nib.load("out/t4_refm.nii.gz").dataobj
        test_img = nib.load("out/t4_test_deformed.nii.gz").dataobj
        np.testing.assert_allclose(ref_img, test_img, atol=1e-10)

    def test4_affine_only_ssc_seg(self):
        ref_seg = nib.load("out/t4_refs.nii.gz").dataobj
        test_seg = nib.load("out/t4_test_deformed_seg.nii.gz").dataobj
        np.testing.assert_allclose(ref_seg, test_seg, atol=1e-10)


    def test5_non_lin_only_lcc_img(self):
        subprocess.call(
            shlex.split("./bin/wbirLCC -F fixed_img.nii -M moving_img.nii -S moving_seg.nii -O out/t5_ref")
        )
        subprocess.call(
            shlex.split("./bin/deedsConvexLCC -F fixed_img.nii.gz -M moving_img.nii.gz  -S moving_seg.nii.gz -O out/t5_test")
        )

        ref_img = nib.load("out/t5_ref_deformed.nii").dataobj
        test_img = nib.load("out/t5_test_deformed.nii.gz").dataobj
        np.testing.assert_allclose(ref_img, test_img, atol=1e-10)

    def test5_non_lin_only_lcc_seg(self):
        ref_seg = nib.load("out/t5_ref_segment.nii").dataobj
        test_seg = nib.load("out/t5_test_deformed_seg.nii.gz").dataobj
        np.testing.assert_allclose(ref_seg, test_seg, atol=1e-10)

    def test6_non_lin_only_ssc_img(self):
        subprocess.call(
            shlex.split("./bin/wbirSSC -F fixed_img.nii -M moving_img.nii -O out/t6_ref -S moving_seg.nii")
        )
        subprocess.call(
            shlex.split("./bin/deedsConvexSSC -F fixed_img.nii.gz -M moving_img.nii.gz -O out/t6_test -S moving_seg.nii.gz")
        )

        ref_img = nib.load("out/t6_ref_deformed.nii").dataobj
        test_img = nib.load("out/t6_test_deformed.nii.gz").dataobj
        np.testing.assert_allclose(ref_img, test_img, atol=1e-10)

    def test6_non_lin_only_ssc_seg(self):
        ref_seg = nib.load("out/t6_ref_segment.nii").dataobj
        test_seg = nib.load("out/t6_test_deformed_seg.nii.gz").dataobj
        np.testing.assert_allclose(ref_seg, test_seg, atol=1e-10)


    def test7_full_lcc_img(self):

        #
        # Reference
        #
        # Transform img aff + seg aff
        subprocess.call(
            shlex.split("./bin/applyBCVfloat -A res/affine_matrix.txt -M moving_img.nii.gz -O res/no_flow -D out/t7_ref_intermedm.nii.gz")
        )
        subprocess.call(
            shlex.split("./bin/applyBCV -A res/affine_matrix.txt -M moving_seg.nii.gz -O res/no_flow -D out/t7_ref_intermeds.nii.gz")
        )
        # gunzip
        subprocess.call(
            shlex.split("gunzip out/t7_ref_intermedm.nii.gz")
        )
        subprocess.call(
            shlex.split("gunzip out/t7_ref_intermeds.nii.gz")
        )
        # Transform non-linear img + seg
        subprocess.call(
            shlex.split("./bin/wbirLCC -F fixed_img.nii -M out/t7_ref_intermedm.nii -O out/t7_ref -S out/t7_ref_intermeds.nii")
        )

        #
        # Test
        #
        subprocess.call(
            shlex.split("./bin/deedsConvexLCC -A res/affine_matrix.txt -F fixed_img.nii.gz -M moving_img.nii.gz -O out/t7_test -S moving_seg.nii.gz")
        )

        ref_img = nib.load("out/t7_ref_deformed.nii").dataobj
        test_img = nib.load("out/t7_test_deformed.nii.gz").dataobj
        np.testing.assert_allclose(ref_img, test_img, atol=1e-10)

    def test7_full_lcc_seg(self):
        ref_seg = nib.load("out/t7_ref_segment.nii").dataobj
        test_seg = nib.load("out/t7_test_deformed_seg.nii.gz").dataobj
        np.testing.assert_allclose(ref_seg, test_seg, atol=1e-10)


    def test8_full_ssc_img(self):

        #
        # Reference
        #
        # Transform img aff + seg aff
        subprocess.call(
            shlex.split("./bin/applyBCVfloat -A res/affine_matrix.txt -M moving_img.nii.gz -O res/no_flow -D out/t8_ref_intermedm.nii.gz")
        )
        subprocess.call(
            shlex.split("./bin/applyBCV -A res/affine_matrix.txt -M moving_seg.nii.gz -O res/no_flow -D out/t8_ref_intermeds.nii.gz")
        )
        # gunzip
        subprocess.call(
            shlex.split("gunzip out/t8_ref_intermedm.nii.gz")
        )
        subprocess.call(
            shlex.split("gunzip out/t8_ref_intermeds.nii.gz")
        )
        # Transform non-linear img + seg
        subprocess.call(
            shlex.split("./bin/wbirSSC -F fixed_img.nii -M out/t8_ref_intermedm.nii -O out/t8_ref -S out/t8_ref_intermeds.nii")
        )

        #
        # Test
        #
        subprocess.call(
            shlex.split("./bin/deedsConvexSSC -A res/affine_matrix.txt -F fixed_img.nii.gz -M moving_img.nii.gz -O out/t8_test -S moving_seg.nii.gz")
        )

        ref_img = nib.load("out/t8_ref_deformed.nii").dataobj
        test_img = nib.load("out/t8_test_deformed.nii.gz").dataobj
        np.testing.assert_allclose(ref_img, test_img, atol=1e-10)

    def test8_full_ssc_seg(self):
        ref_seg = nib.load("out/t8_ref_segment.nii").dataobj
        test_seg = nib.load("out/t8_test_deformed_seg.nii.gz").dataobj
        np.testing.assert_allclose(ref_seg, test_seg, atol=1e-10)

if __name__ == '__main__':
    unittest.main()