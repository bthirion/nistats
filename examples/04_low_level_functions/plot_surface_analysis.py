"""
Surface-based analysis of fMRI data.
This examples
1. Projects an fMRI dataset onto the surface
2. Analyses the data on the surface
3. Generates a plot of the data on the surface
"""

#########################################################################
# Prepare data and analysis parameters
# -------------------------------------
# Prepare timing
t_r = 2.4
slice_time_ref = 0.5

# Prepare data
from nistats import datasets
import pandas as pd
data = datasets.fetch_localizer_first_level()
paradigm_file = data.paradigm
paradigm = pd.read_csv(paradigm_file, sep=' ', header=None, index_col=None)
paradigm.columns = ['session', 'trial_type', 'onset']


#########################################################################
# Projection to the surface
# -------------------------

from nilearn.datasets import fetch_surf_fsaverage5
fsaverage = fetch_surf_fsaverage5()
from nilearn import surface
texture = surface.vol_to_surf(data.epi_img, fsaverage.pial_right)

#########################################################################
# Perform first level analysis
# ----------------------------
# Set up design matrix
import numpy as np
from nistats.design_matrix import make_design_matrix
n_scans = texture.shape[1]
frame_times = (slice_time_ref + np.arange(n_scans)) * t_r
X = make_design_matrix(
    frame_times, paradigm, hrf_model='glover + derivative')

# fit the GLM
from nistats.first_level_model import run_glm
labels, results = run_glm(texture.T, X.values, noise_model='ar1')

#########################################################################
# Estimate contrasts
# ------------------
# Specify the contrasts

design_matrix = X
contrast_matrix = np.eye(design_matrix.shape[1])
contrasts = dict([(column, contrast_matrix[i])
                  for i, column in enumerate(design_matrix.columns)])

contrasts["audio"] = contrasts["clicDaudio"] + contrasts["clicGaudio"] +\
    contrasts["calculaudio"] + contrasts["phraseaudio"]
contrasts["video"] = contrasts["clicDvideo"] + contrasts["clicGvideo"] + \
    contrasts["calculvideo"] + contrasts["phrasevideo"]
contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]

#########################################################################
# Short list of more relevant contrasts
contrasts = {
    "left-right": (contrasts["clicGaudio"] + contrasts["clicGvideo"]
                   - contrasts["clicDaudio"] - contrasts["clicDvideo"]),
    "Horizontal-Vertical": contrasts["damier_H"] - contrasts["damier_V"],
    "audio-video": contrasts["audio"] - contrasts["video"],
    "video-audio": -contrasts["audio"] + contrasts["video"],
    "computation-sentences": (contrasts["computation"] -
                              contrasts["sentences"]),
    "reading-visual": contrasts["phrasevideo"] - contrasts["damier_H"]
    }

#########################################################################
# contrast estimation
from nistats.contrasts import compute_contrast
from nilearn import plotting


for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    z_values = compute_contrast(labels, results, contrast_val).z_score()

    # Create snapshots of the contrasts
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, z_values, hemi='right', title=contrast_id,
        threshold=3., bg_map=fsaverage.sulc_right, cmap='cold_hot')

plotting.show()
