"""
Second-level Mixed-effects model: one sample test
========================================

Full step-by-step example of performing a second-level analysis (one-sample test) using a mixed effects model, assuming that first-level estimated effects and variance maps are provided for all subjects.

More specifically:

1. A bunch of effects and variance maps are downloaded
2. A classical one-sample  t-test is carried out on effects maps
3. For comparison, a mixed effects test taking into account effects and variance is run

We focus on 13 subjects performing the (slightly redesigned) relational task of the Human Connectome Project dataset. These data have been acquired from the Individual Brain CHarting collection

todo:
* remove masker
* upload data onto osf

"""

#########################################################################
# Fetch dataset
# --------------
# We download a list of left vs right button press contrasts from a
# localizer dataset. Note that we fetc individual t-maps that represent the Bold activity estimate divided by the uncertainty about this estimate. 
import glob
import shutil
effect_imgs = sorted(glob.glob(
    '/tmp/relational-match_maps/*_effect_relational-match.nii.gz'))
variance_imgs = sorted(glob.glob(
    '/tmp/relational-match_maps/*_variance_relational-match.nii.gz'))

############################################################################
# Estimate second level model
# ---------------------------
# We define the input maps and the design matrix for the second level model
# and fit it.
import pandas as pd
design_matrix = pd.DataFrame([1] * len(effect_imgs),
                             columns=['intercept'])

############################################################################
# Model specification and fit
from nistats.second_level_model import SecondLevelModel
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(effect_imgs,
                                            design_matrix=design_matrix)
z_map = second_level_model.compute_contrast(output_type='z_score')

############################################################################
# Compute threshold for family-wise error rate (fwer) control
from nistats.thresholding import map_threshold
_, threshold = map_threshold(z_map, level=.05,height_control='bonferroni' )

from nilearn import plotting
plotting.plot_glass_brain(
    z_map, threshold=threshold, colorbar=True, display_mode='z', plot_abs=False,
    title='random effects test (p<0.05, fwer-corrected)')

###########################################################################
# Now, do the same with a mixed effects model that takes into acount first-level effects undertainty estimate
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_strategy='background').fit(effect_imgs)

# the function below computes mixed effects + 1000 permutations to get
# the distribution under the null

from nistats.mixed_effects_model import mixed_effects_likelihood_ratio_test
beta, variance, z_map, max_diff_z = mixed_effects_likelihood_ratio_test(
    masker, effect_imgs, variance_imgs, design_matrix, contrast=[[1]])

import numpy as np
threshold = np.percentile(max_diff_z, 95)
plotting.plot_glass_brain(
    z_map, threshold=threshold, colorbar=True, display_mode='z', plot_abs=False,
    title='mixed effects test (p<0.05, fwer-corrected)')


plotting.show()
