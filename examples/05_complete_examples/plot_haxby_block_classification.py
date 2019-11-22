"""
Decoding of a dataset after glm fit for signal extraction
=========================================================


Full step-by-step example of fitting a GLM to perform a decoding experiment.
We use the data from one subject of the celebrated Haxby dataset

More specifically:

1. Download the Haxby dataset
2. Extract the information to generate a glm representing the blocks of stimuli
3. Analyze the decoding performance using a classifier

To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use the Jupyter notebook.

.. contents:: **Contents**
    :local:
    :depth: 1
"""

##############################################################################
# Fetch example KHaxby dataset
# ----------------------------
# We download the Haxby dataset
# This is a study of visual obkect category representation
from nilearn import datasets

# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()

# repetition has to be known
TR = 2.5 

#############################################################################
# Load the behavioral data
# -------------------------
import pandas as pd

# Load target information as string and give a numerical identifier to each
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
conditions = behavioral['labels'].values

# Record these as an array of sessions
sessions = behavioral['chunks'].values
unique_session = behavioral['chunks'].unique()

# fMRI data: a unique file for all sessions
func_filename = haxby_dataset.func[0]
    
#############################################################################
# Build a proper event structure for each session
# -----------------------------------------------
import numpy as np
events = {}
# events will take  the form of a dictionary of Dataframes, one per session
for session in unique_session:
    # get the condition label per session
    conditions_session = conditions[sessions == session]
    # get the number of scans per session, then the corresponding
    # vector of frame times
    n_scans = len(conditions_session)
    frame_times = TR * np.arange(n_scans)
    # each event last the full TR
    duration = TR * np.ones(n_scans)
    # Define the events object
    events_ = pd.DataFrame(
        {'onset': frame_times, 'trial_type': conditions_session, 'duration': duration })
    # remove the rest condition and insert into the dictionary
    events[session] = events_[events_.trial_type != 'rest']

##############################################################################
# Instantiate and run FirstLevelModel
# -----------------------------------    
from nistats.first_level_model import FirstLevelModel
from nilearn.image import index_img

# we are going to generate a list of z-maps together with their session and condition index
z_maps = []
condition_idx = []
session_idx = []

# Instantiate the glm
glm = FirstLevelModel(t_r=TR,
                      mask=haxby_dataset.mask,
                      high_pass=.008,
                      smoothing_fwhm=4,
                      memory="nilearn_cache")

for session in np.unique(sessions):
    print(session)
    fmri_session = index_img(func_filename, sessions == session)
    glm.fit(fmri_session, events=events[session])

    # set up contrasts: one per condition
    conditions = events[session].trial_type.unique()
    for condition in conditions:
        z_maps.append(glm.compute_contrast(condition))
        condition_idx.append(condition)
        session_idx.append(session)
        
#############################################################################
# Transform the maps to an array of values
# ----------------------------------------
from nilearn.input_data import NiftiMasker

# no need to standardize or smooth the data
masker = NiftiMasker(mask_img=haxby_dataset.mask, memory="nilearn_cache", memory_level=1)
X = masker.fit_transform(z_maps)

#############################################################################
# Build the decoder
# ------------------
# Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
from sklearn.svm import SVC
svc = SVC(kernel='linear')

# Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. When doing full-brain analysis, it is better to use
# SelectPercentile, keeping 5% of voxels
# (because it is independent of the resolution of the data).
from sklearn.feature_selection import SelectPercentile, f_classif
feature_selection = SelectPercentile(f_classif, percentile=5)

# We have our classifier (SVC), our feature selection (SelectPercentile),and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

#############################################################################
# Obtain prediction scores via cross validation
# -----------------------------------------------
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

# Define the cross-validation scheme used for validation.
# Here we use a LeaveOneGroupOut cross-validation on the session group
# which corresponds to a leave-one-session-out
cv = LeaveOneGroupOut()

# Compute the prediction accuracy for the different folds (i.e. session)
cv_scores = cross_val_score(anova_svc, X, condition_idx, cv=cv, groups=session_idx)

# Return the corresponding mean prediction accuracy
classification_accuracy = cv_scores.mean()

# Print the results
print("Classification accuracy: %.4f / Chance level: %f" %
      (classification_accuracy, 1. / len(np.unique(condition_idx))))
# Classification accuracy:  0.375 / Chance level: 0.125



plotting.show()
