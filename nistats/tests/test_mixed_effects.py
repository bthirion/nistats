from nose.tools import (assert_true,
                        assert_equal,
                        assert_raises,
                        )
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           )
from nilearn._utils.exceptions import DimensionError
import pandas as pd
import numpy as np
from nibabel import Nifti1Image
from nilearn.image import math_img
from nibabel.tmpdirs import InTemporaryDirectory
from nistats.mixed_effects_model import mixed_effects_likelihood_ratio_stat
from nistats._utils.testing import _write_fake_fmri_data


def test_mixed_effects_lrt():
    with InTemporaryDirectory():
        n_subjects = 10
        shapes = ((3, 4, 5, n_subjects),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        effects = FUNCFILE[0]
        variance = math_img('.1 * i ** 2', i=effects)
        design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)}) 
        
        # one-sample test
        n_perm = 1
        for contrast in [np.array([[1]])]:
            beta, variance_, z_map, max_diff_z =\
                mixed_effects_likelihood_ratio_stat(
                    effects, variance, design_matrix, contrast, mask_img=mask,
                    n_perm=n_perm, n_iter=5, n_jobs=1)
            for img in [beta, variance_, z_map]:
                assert_true(isinstance(img, Nifti1Image))
                assert_equal(len(max_diff_z), n_perm)
                assert_true(max_diff_z[0] > 0)
                assert_true((variance.get_data() > 0).all())
                assert_almost_equal(z_map.get_data().mean(), 0, 0)
        # different design matrix
        design_matrix = pd.DataFrame(
            {'intercept': np.ones(n_subjects),
             'plop': np.random.rand(n_subjects)})
        n_perm = 2
        for contrast in (np.array([[0, 1]]).T, np.array([[1, -1]]).T):
            beta, variance_, z_map, max_diff_z =\
                    mixed_effects_likelihood_ratio_stat(
                    effects, variance, design_matrix, contrast, mask_img=mask,
                    n_perm=n_perm, n_iter=5, n_jobs=1)
            assert_equal(len(max_diff_z), n_perm)
            assert_true(max_diff_z[0] > 0)
            assert_true((variance.get_data() > 0).all())
            assert_almost_equal(z_map.get_data().mean(), 0, 0)

        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del FUNCFILE


def test_fmri_inputs():
    # Test processing of FMRI inputs
    with InTemporaryDirectory():
        n_subjects = 10
        shapes = ((3, 4, 5, n_subjects),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        effects = FUNCFILE[0]
        variance = math_img('.1 * i ** 2', i=effects)
        design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)}) 
        
        # one-sample test
        n_perm = 1
        contrast  = np.array(([1]))
        ## Fail when contrast is None
        assert_raises(ValueError, mixed_effects_likelihood_ratio_stat,
                      effects, variance, design_matrix, None, mask_img='a')
        # Fail when design matrix is not correct size
        design_matrix_ = pd.DataFrame({'intercept': np.ones(n_subjects + 1)}) 
        
        assert_raises(ValueError, mixed_effects_likelihood_ratio_stat,
                      effects, variance, design_matrix_, contrast, mask_img=mask)

        shapes = ((3, 4, 5, n_subjects + 1),)
        # Fail when data sizes are not consistent
        _, FUNCFILE_, _ = _write_fake_fmri_data(shapes)
        assert_raises(ValueError, mixed_effects_likelihood_ratio_stat,
                      FUNCFILE_[0], variance, design_matrix, contrast, mask_img=mask)
        assert_raises(DimensionError, mixed_effects_likelihood_ratio_stat,
                      [FUNCFILE], variance, design_matrix, contrast, mask_img=mask)
        assert_raises(DimensionError, mixed_effects_likelihood_ratio_stat,
                      FUNCFILE, [variance], design_matrix, contrast, mask_img=mask)
