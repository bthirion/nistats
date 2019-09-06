"""Simple example of fixed effects fMRI model fitting
================================================

This example illustrates how to 

For details on the data, please see:

Dehaene-Lambertz G, Dehaene S, Anton JL, Campagne A, Ciuciu P, Dehaene
G, Denghien I, Jobert A, LeBihan D, Sigman M, Pallier C, Poline
JB. Functional segregation of cortical language areas by sentence
repetition. Hum Brain Mapp. 2006: 27:360--371.
http://www.pubmedcentral.nih.gov/articlerender.fcgi?artid=2653076#R11

Please see `plot_fiac_analysis.py` example for details.  The main
difference is that the fixed-effects model is run explicitly here,
after GLM fitting on two sessions.

"""

###############################################################################
# Create a write directory to work
# it will be a 'results' subdirectory of the current directory.
from os import mkdir, path, getcwd
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

#########################################################################
# Prepare data and analysis parameters
# --------------------------------------
# 
# Note that there are two sessions

from nistats import datasets
data = datasets.fetch_fiac_first_level()
fmri_img = [data['func1'], data['func2']]

#########################################################################
# Create a mean image for plotting purpose
from nilearn.image import mean_img
mean_img_ = mean_img(fmri_img[0])

#########################################################################
# The design matrices were pre-computed, we simply put them in a list of DataFrames
design_files = [data['design_matrix1'], data['design_matrix2']]
import pandas as pd
import numpy as np
design_matrices = [pd.DataFrame(np.load(df)['X']) for df in design_files]

#########################################################################
# GLM estimation
# ----------------------------------
# GLM specification. Note that the mask was provided in the dataset. So we use it.

from nistats.first_level_model import FirstLevelModel
fmri_glm = FirstLevelModel(mask_img=data['mask'], smoothing_fwhm=5,
                           minimize_memory=True)

#########################################################################
# Compute fixed effects of the two runs and compute related images
# For this, we first define the contrasts as we would do for a single session
n_columns = design_matrices[0].shape[1]

def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

#########################################################################
# Contrast specification

contrast_id = 'DSt_minus_SSt'
contrast_val = pad_vector([-1, -1, 1, 1], n_columns)

#########################################################################
# Statistics for the first session
from nilearn import plotting
cut_coords = [-129, -126, 49]

fmri_glm = fmri_glm.fit(fmri_img[0], design_matrices=design_matrices[0])
dic1 = fmri_glm.compute_contrast(contrast_val, output_type='all')
plotting.plot_stat_map(
    dic1['z_score'], bg_img=mean_img_, threshold=3.0, cut_coords=cut_coords,
    title='%s, first session' % contrast_id)

#########################################################################
# Statistics for the second session

fmri_glm = fmri_glm.fit(fmri_img[1], design_matrices=design_matrices[1])
dic2 = fmri_glm.compute_contrast(contrast_val, output_type='all')
plotting.plot_stat_map(
    dic2['z_score'], bg_img=mean_img_, threshold=3.0, cut_coords=cut_coords,
    title='%s, second session' % contrast_id)

#########################################################################
# Fixed effects statistics
from nistats.contrasts import fixed_effects_img

contrast_imgs = [dic1['effect_size'], dic2['effect_size']]
variance_imgs = [dic1['effect_variance'], dic2['effect_variance']]

ffx_contrast, ffx_variance, ffx_stat = fixed_effects_img(
    contrast_imgs, variance_imgs, data['mask'])
plotting.plot_stat_map(
    ffx_stat, bg_img=mean_img_, threshold=3.0, cut_coords=cut_coords,
    title='%s, fixed effects' % contrast_id )

#########################################################################
# Not unexpectedly, the fixed effects version looks displays higher peaks than the input sessions. Computing fixed effects enhances the signal-to-noise ratio of the resulting brain maps
# Note however that, technically, the output maps of the fixed effects map is a t statistic (not a z statistic)

plotting.show()
