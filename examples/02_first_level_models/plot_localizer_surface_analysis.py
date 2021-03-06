"""Example of surface-based first-level analysis
=============================================

Full step-by-step example of fitting a GLM to experimental data
sampled on the cortical surface and visualizing the results.

More specifically:

1. A sequence of fMRI volumes are loaded
2. fMRI data are projected onto a reference cortical surface (the freesurfer template, fsaverage)
3. A design matrix describing all the effects related to the data is computed
4. A GLM is applied to the dataset (effect/covariance, then contrast estimation)

The result of the analysis are statistical maps that are defined on
the brain mesh. We display them using Nilearn capabilities.

The projection of fMRI data onto a given brain mesh requires that both
are initially defined in the same space.

* The functional data should be coregistered to the anatomy from which the mesh was obtained.

* Another possibility, used here, is to project the normalized fMRI data to an MNI-coregistered mesh, such as fsaverage.

The advantage of this second approach is that it makes it easy to run
second-level analyses on the surface. On the other hand, it is
obviously less accurate than using a subject-tailored mesh.

"""

#########################################################################
# Prepare data and analysis parameters
# -------------------------------------
# Prepare timing parameters
t_r = 2.4
slice_time_ref = 0.5

#########################################################################
# Prepare data
# First the volume-based fMRI data.
from nistats.datasets import fetch_localizer_first_level
data = fetch_localizer_first_level()
fmri_img = data.epi_img

#########################################################################
# Second the experimental paradigm.
events_file = data['events']
import pandas as pd
events = pd.read_table(events_file)

#########################################################################
# Project the fMRI image to the surface
# -------------------------------------
#
# For this we need to get a mesh representing the geometry of the
# surface.  we could use an individual mesh, but we first resort to a
# standard mesh, the so-called fsaverage5 template from the Freesurfer
# software.

import nilearn
fsaverage = nilearn.datasets.fetch_surf_fsaverage5()

#########################################################################
# The projection function simply takes the fMRI data and the mesh.
# Note that those correspond spatially, are they are bothin MNI space.
from nilearn import surface
texture = surface.vol_to_surf(fmri_img, fsaverage.pial_right)

#########################################################################
# Perform first level analysis
# ----------------------------
#
# This involves computing the design matrix and fitting the model.
# We start by specifying the timing of fMRI frames

import numpy as np
n_scans = texture.shape[1]
frame_times = t_r * (np.arange(n_scans) + .5)

#########################################################################
# Create the design matrix
#
# We specify an hrf model containing Glover model and its time derivative
# the drift model is implicitly a cosine basis with period cutoff 128s.
from nistats.design_matrix import make_first_level_design_matrix
design_matrix = make_first_level_design_matrix(frame_times,
                                               events=events,
                                               hrf_model='glover + derivative'
                                               )

#########################################################################
# Setup and fit GLM.
# Note that the output consists in 2 variables: `labels` and `fit`
# `labels` tags voxels according to noise autocorrelation.
# `estimates` contains the parameter estimates.
# We keep them for later contrast computation.

from nistats.first_level_model import run_glm
labels, estimates = run_glm(texture.T, design_matrix.values)

#########################################################################
# Estimate contrasts
# ------------------
# Specify the contrasts
# For practical purpose, we first generate an identity matrix whose size is
# the number of columns of the design matrix
contrast_matrix = np.eye(design_matrix.shape[1])

#########################################################################
# first create basic contrasts
basic_contrasts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])

#########################################################################
# add some intermediate contrasts
basic_contrasts["audio"] = (basic_contrasts["clicDaudio"]
                            + basic_contrasts["clicGaudio"]
                            + basic_contrasts["calculaudio"]
                            + basic_contrasts["phraseaudio"]
                            )
basic_contrasts["video"] = (basic_contrasts["clicDvideo"]
                            + basic_contrasts["clicGvideo"]
                            + basic_contrasts["calculvideo"]
                            + basic_contrasts["phrasevideo"]
                            )
basic_contrasts["computation"] = (basic_contrasts["calculaudio"]
                                  + basic_contrasts["calculvideo"]
                                  )
basic_contrasts["sentences"] = (basic_contrasts["phraseaudio"]
                                + basic_contrasts["phrasevideo"]
                                )

#########################################################################
# Finally make a dictionary of more relevant contrasts
#
# * "left - right button press" probes motor activity in left versus right button presses
# * "audio - video" probes the difference of activity between listening to some content or reading the same type of content (instructions, stories)
# * "computation - sentences" looks at the activity when performing a mental comptation task  versus simply reading sentences.
#
# Of course, we could define other contrasts, but we keep only 3 for simplicity.

contrasts = {
    "left - right button press": (basic_contrasts["clicGaudio"]
                                  + basic_contrasts["clicGvideo"]
                                  - basic_contrasts["clicDaudio"]
                                  - basic_contrasts["clicDvideo"]
                                  ),
    "audio - video": basic_contrasts["audio"] - basic_contrasts["video"],
    "computation - sentences": (basic_contrasts["computation"] -
                                basic_contrasts["sentences"]
                                )
    }

#########################################################################
# contrast estimation
from nistats.contrasts import compute_contrast
from nilearn import plotting

#########################################################################
# iterate over contrasts
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % i out of %i: %s, right hemisphere' %
          (index + 1, len(contrasts), contrast_id))
    # compute contrast-related statistics
    contrast = compute_contrast(labels, estimates, contrast_val,
                                contrast_type='t')
    # we present the Z-transform of the t map
    z_score = contrast.z_score()
    # we plot it on the surface, on the inflated fsaverage mesh,
    # together with a suitable background to give an impression
    # of the cortex folding.
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, z_score, hemi='right',
        title=contrast_id, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_right)

#########################################################################
# Analysing the left hemisphere
# -----------------------------
#
# Note that it requires little additional code!

#########################################################################
# Project the fMRI data to the mesh
texture = surface.vol_to_surf(fmri_img, fsaverage.pial_left)

#########################################################################
# Estimate the General Linear Model
labels, estimates = run_glm(texture.T, design_matrix.values)

#########################################################################
# Create contrast-specific maps
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % i out of %i: %s, left hemisphere' %
          (index + 1, len(contrasts), contrast_id))
    # compute contrasts
    contrast = compute_contrast(labels, estimates, contrast_val,
                                contrast_type='t')
    z_score = contrast.z_score()
    # Plot the result
    plotting.plot_surf_stat_map(
        fsaverage.infl_left, z_score, hemi='left',
        title=contrast_id, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_left)

plotting.show()
