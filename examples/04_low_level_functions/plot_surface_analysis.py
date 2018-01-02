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
# fmri_img = data.epi_img


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
# Setup and fit GLM
from nistats.first_level_model import FirstLevelModel
first_level_model = FirstLevelModel(t_r, slice_time_ref,
                                    hrf_model='glover + derivative')
first_level_model = first_level_model.fit(texture, paradigm)

#########################################################################
# Estimate contrasts
# ------------------
# Specify the contrasts
design_matrix = first_level_model.design_matrices_[0]
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
    "H-V": contrasts["damier_H"] - contrasts["damier_V"],
    "audio-video": contrasts["audio"] - contrasts["video"],
    "video-audio": -contrasts["audio"] + contrasts["video"],
    "computation-sentences": (contrasts["computation"] -
                              contrasts["sentences"]),
    "reading-visual": contrasts["phrasevideo"] - contrasts["damier_H"]
    }

#########################################################################
# contrast estimation
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    z_map = first_level_model.compute_contrast(contrast_val,
                                               output_type='z_score')

    # Create snapshots of the contrasts
    display = plotting.plot_stat_map(z_map, display_mode='z',
                                     threshold=3.0, title=contrast_id)

plotting.show()
