"""Extension of second_level_model to handle mixed effects,
ie two levels of variance corresponding to
intra- and inter-subject variability respectively.

Importantly, the first level variance is assumed to be known (and kept fixed),
and only population parameters are estimated.

The focus is put on inference: Getting statistics with correct
probabilistic control. In the absence of a correct analytical model, we
provide a permutation test.

Author: Bertrand Thirion, 2019

"""
import numpy as np
from nistats.second_level_model import SecondLevelModel
from joblib import Parallel, delayed


def _mixed_log_likelihood(data, mean, V1, V2):
    """ return the log-likelighood of the data under a composite variance model
    """
    total_variance = V1 + V2
    logl = np.sum(((data - mean) ** 2) / total_variance, 0)
    logl += np.sum(np.log(total_variance), 0)
    logl += np.log(2 * np.pi) * data.shape[0]
    logl *= (- 0.5)
    return logl


def _em_step(Y, Y_, V1, V2, X, pinv_X):
    """ One-pass estimate of effects and variance estimates
    (corresponds to one step of EM).
    """
    # E step
    precision = 1. / (V1 + V2)
    _Y = precision * (V2 * Y + V1 * Y_)
    variance_ = (V1 * V2 * precision).mean(0)

    # M step
    beta_ = np.dot(pinv_X, _Y)
    Y_ = np.dot(X, beta_)
    V2_ = np.mean((Y_ - _Y) ** 2, 0) + variance_
    return beta_, Y_, V2_


def mixed_model_inference(X, Y, V1, n_iter=5, verbose=0):
    """ Run an EM algorithm to perform mixed_model_inference
    """
    pinv_X = np.linalg.pinv(X)
    beta_ = np.dot(pinv_X, Y)
    Y_ = np.dot(X, beta_)
    V2_ = np.mean((Y - Y_) ** 2, 0)

    if verbose:
        logl = _mixed_log_likelihood(Y, Y_, V1, V2_)
        print('Average log-likelihood: ', logl.mean())

    for i in range(n_iter):
        beta_, Y_, V2_ = _em_step(Y, Y_, V1, V2_, X, pinv_X)
        if verbose:
            logl_ = _mixed_log_likelihood(Y, Y_, V1, V2_)
            if (logl_ < (logl - 100 * np.finfo(float).eps)).any():
                raise ValueError('The log-likelihood cannot decrease')
            logl = logl_
            print('Iteration %d, average log-likelihood: %f' % (
                i, logl_.mean()))

    if not verbose:
        # need to return loglikelihood
        logl_ = _mixed_log_likelihood(Y, Y_, V1, V2_)
    return beta_, V2_, logl_


def _randomize_design(X):
    """small utility to randomize the rows of the design matrix"""
    X_ = X.copy()
    np.random.shuffle(X_)
    if (X_ == X).all():
        sign_swap = 2 * (np.random.rand(*X.shape) > .5) - 1
        X_ *= sign_swap
    return X_


def _permuted_max_llr(X, contrast, Y, V1, n_iter):
    """utility function that randomizes the design and computes the maximum
    log-likelihood ratio over the data."""
    X_ = _randomize_design(X)
    X_null_ = X - np.dot(np.dot(X_, contrast), np.linalg.pinv(contrast))
    _, _, log_likelihood_ = mixed_model_inference(X_, Y, V1, n_iter=n_iter)
    _, _, log_likelihood_null_ = mixed_model_inference(
        X_null_, Y, V1, n_iter=n_iter)
    log_likelihood_ratio_ = np.maximum(
        log_likelihood_ - log_likelihood_null_, 0)
    return log_likelihood_ratio_.max()


def permuted_mixed_effects(
        tested_var, confoudning_vars, target_vars, variance_target_vars,
        model_intercept=True,
        n_perm=1000, two_sided_test=True,
        random_state=42, n_jobs=1, verbose=0):
    """ 


    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    # check n_jobs (number of CPUs)
    if n_jobs == 0:  # invalid according to joblib's conventions
        raise ValueError("'n_jobs == 0' is not a valid choice. "
                         "Please provide a positive number of CPUs, or -1 "
                         "for all CPUs, or a negative number (-i) for "
                         "'all but (i-1)' CPUs (joblib conventions).")
    elif n_jobs < 0:
        n_jobs = max(1, joblib.cpu_count() - int(n_jobs) + 1)
    else:
        n_jobs = min(n_jobs, joblib.cpu_count())
    # make target_vars F-ordered to speed-up computation
    if target_vars.ndim != 2:
        raise ValueError("'target_vars' should be a 2D array. "
                         "An array with %d dimension%s was passed"
                         % (target_vars.ndim,
                            "s" if target_vars.ndim > 1 else ""))
    target_vars = np.asfortranarray(target_vars)  # efficient for chunking
    n_descriptors = target_vars.shape[1]

    # check explanatory variates dimensions
    if tested_vars.ndim == 1:
        tested_vars = np.atleast_2d(tested_vars).T
    n_samples, n_regressors = tested_vars.shape

    # check if explanatory variates is intercept (constant) or not
    if (n_regressors == 1 and np.unique(tested_vars).size == 1):
        intercept_test = True
    else:
        intercept_test = False

    # optionally add intercept
    if model_intercept and not intercept_test:
        if confounding_vars is not None:
            confounding_vars = np.hstack(
                (confounding_vars, np.ones((n_samples, 1))))
        else:
            confounding_vars = np.ones((n_samples, 1))

    X = np.vstack((tested_vars, confounding_vars))
    contrast = np.zeros(X.shape[1])
    contrast[:tested_vars.shape[1]] = 1
    Y = target_vars
    V1 = variance_target_vars

    
    """
    #-------------------------------------------------------------------
    ### OLS regression on original data
    if confounding_vars is not None:
        # step 1: extract effect of covars from target vars
        covars_orthonormalized = orthonormalize_matrix(confounding_vars)
        if not covars_orthonormalized.flags['C_CONTIGUOUS']:
            # useful to developer
            warnings.warn('Confounding variates not C_CONTIGUOUS.')
            covars_orthonormalized = np.ascontiguousarray(
                covars_orthonormalized)
        targetvars_normalized = normalize_matrix_on_axis(
            target_vars).T  # faster with F-ordered target_vars_chunk
        if not targetvars_normalized.flags['C_CONTIGUOUS']:
            # useful to developer
            warnings.warn('Target variates not C_CONTIGUOUS.')
            targetvars_normalized = np.ascontiguousarray(targetvars_normalized)
        beta_targetvars_covars = np.dot(targetvars_normalized,
                                        covars_orthonormalized)
        targetvars_resid_covars = targetvars_normalized - np.dot(
            beta_targetvars_covars, covars_orthonormalized.T)
        targetvars_resid_covars = normalize_matrix_on_axis(
            targetvars_resid_covars, axis=1)
        # step 2: extract effect of covars from tested vars
        testedvars_normalized = normalize_matrix_on_axis(tested_vars.T, axis=1)
        beta_testedvars_covars = np.dot(testedvars_normalized,
                                        covars_orthonormalized)
        testedvars_resid_covars = testedvars_normalized - np.dot(
            beta_testedvars_covars, covars_orthonormalized.T)
        testedvars_resid_covars = normalize_matrix_on_axis(
            testedvars_resid_covars, axis=1).T.copy()
    else:
        targetvars_resid_covars = normalize_matrix_on_axis(target_vars).T
        testedvars_resid_covars = normalize_matrix_on_axis(tested_vars).copy()
        covars_orthonormalized = None
    # check arrays contiguousity (for the sake of code efficiency)
    if not targetvars_resid_covars.flags['C_CONTIGUOUS']:
        # useful to developer
        warnings.warn('Target variates not C_CONTIGUOUS.')
        targetvars_resid_covars = np.ascontiguousarray(targetvars_resid_covars)
    if not testedvars_resid_covars.flags['C_CONTIGUOUS']:
        # useful to developer
        warnings.warn('Tested variates not C_CONTIGUOUS.')
        testedvars_resid_covars = np.ascontiguousarray(testedvars_resid_covars)
    # step 3: original regression (= regression on residuals + adjust t-score)
    # compute t score for original data
    scores_original_data = _t_score_with_covars_and_normalized_design(
        testedvars_resid_covars, targetvars_resid_covars.T,
        covars_orthonormalized)
    if two_sided_test:
        sign_scores_original_data = np.sign(scores_original_data)
        scores_original_data = np.fabs(scores_original_data)

    ### Permutations
    # parallel computing units perform a reduced number of permutations each
    if n_perm > n_jobs:
        n_perm_chunks = np.asarray([n_perm / n_jobs] * n_jobs, dtype=int)
        n_perm_chunks[-1] += n_perm % n_jobs
    elif n_perm > 0:
        warnings.warn('The specified number of permutations is %d and '
                      'the number of jobs to be performed in parallel has '
                      'set to %s. This is incompatible so only %d jobs will '
                      'be running. You may want to perform more permutations '
                      'in order to take the most of the available computing '
                      'ressources.' % (n_perm, n_jobs, n_perm))
        n_perm_chunks = np.ones(n_perm, dtype=int)
    else:  # 0 or negative number of permutations => original data scores only
        if two_sided_test:
            scores_original_data = (scores_original_data
                                    * sign_scores_original_data)
        return np.asarray([]), scores_original_data,  np.asarray([])
    # actual permutations, seeded from a random integer between 0 and maximum
    # value represented by np.int32 (to have a large entropy).
    ret = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(_permuted_ols_on_chunk)(
            scores_original_data, testedvars_resid_covars,
            targetvars_resid_covars.T, covars_orthonormalized,
            n_perm_chunk=n_perm_chunk, intercept_test=intercept_test,
            two_sided_test=two_sided_test,
            random_state=rng.random_integers(np.iinfo(np.int32).max - 1))
        for n_perm_chunk in n_perm_chunks)
    # reduce results
    scores_as_ranks_parts, h0_fmax_parts = zip(*ret)
    h0_fmax = np.hstack((h0_fmax_parts))
    scores_as_ranks = np.zeros((n_regressors, n_descriptors))
    for scores_as_ranks_part in scores_as_ranks_parts:
        scores_as_ranks += scores_as_ranks_part
    # convert ranks into p-values
    pvals = (n_perm + 1 - scores_as_ranks) / float(1 + n_perm)

    # put back sign on scores if it was removed in the case of a two-sided test
    # (useful to distinguish between positive and negative effects)
    if two_sided_test:
        scores_original_data = scores_original_data * sign_scores_original_data

    return - np.log10(pvals), scores_original_data.T, h0_fmax[0]

    X_null = X - np.dot(np.dot(X, contrast), np.linalg.pinv(contrast))
    # get parameters estimates for the full model
    beta_, V2, log_likelihood = mixed_model_inference(X, Y, V1, n_iter=n_iter)
    # parameters estimates for the reduced model
    _, _, log_likelihood_null = mixed_model_inference(
        X_null, Y, V1, n_iter=n_iter)
    # compute the log of likrelihood ratio between the two models
    logl_ratio = np.maximum(log_likelihood - log_likelihood_null, 0)
    # convert the permuted log-likelihood to z-scores
    z_ = np.sqrt(2 * logl_ratio) * np.sign(beta_)
            
    max_diff_loglike = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_max_llr)(X, contrast, Y, V1, n_iter)
        for _ in range(n_perm))

    # convert the permuted log-likelihood to z-scores
    max_diff_z = np.sqrt(2 * np.array(sorted(max_diff_loglike)))

    # Generate outputs
    beta = masker.inverse_transform(beta_)
    z_map = masker.inverse_transform(z_)
    second_level_variance = masker.inverse_transform(V2)
    """
    return mixed_effects_likelihood_ratio_test(
        target_vars, variance_target_vars, X, contrast,
        mask_img=None, #  ???
        n_perm=n_perm, n_iter=n_iter, n_jobs=n_jobs)


def mixed_effects_likelihood_ratio_test(
        effects, variance, design_matrix, contrast, mask_img=None,
        n_perm=1000, n_iter=5, n_jobs=1):
    """Compute a second-level contrast given brain maps

    Parameters
    ----------
    effects_maps: niimg,
        set of volumes representing the estimated effects
    variance_maps: niimg,
        set of volumes representing the effects variance
    design_matrix: numpy array,
        second-level design matrix that represents explanatory variables
        across subjects
    contrast: numpy array,
        specification of a second-level contrast to infer upon
    mask_img: niimg, optional,
        Mask image for the analysis
    n_perm: int, optional,
        Number of permutations in the non-parametric test
    n_jobs: int, optional:
        Number of jobs for computation of permutations

    Returns
    -------
    beta: Nifti1Image,
        the group-level map of estimated effect size
    second_level_variance_map: Nifti1Image,
        the group level map of estimated cross-subject variance
    z_map: Nifti1Image
        A stat maps that represents a z-scale map of the statistic;
        it is based on a log-likelihood ratio for the specified contrast.
        the map is signed according to the estimated effect (beta map)
    max_diff_z: numpy array,
        The set of maximal z-values obtained under permutations of the model
        This represents a non-parametric distribution of max(z) under the null.
        According to westfall-Young procedure, quantiles of this provide
        fwer-corrected thresholds. 
    """
    from nilearn.input_data import NiftiMasker
    if mask_img is None:
        masker = NiftiMasker(mask_strategy='background').fit(effects)
    else:
        masker = NiftiMasker(mask_img=mask_img).fit()
    
    Y = masker.transform(effects)
    V1 = masker.transform(variance)

    try:
        contrast = np.atleast2d(contrast).astype(np.float)
    except:
        ValueError(
            'Provided contrast cannot be cast to 2d array')

    if Y.shape != V1.shape:
        raise ValueError(
            'Effects and variance have inconsistent shapes.'
            'Shape of effects is %s, shape of variance is %s'
            % (str(Y.shape), str(V1.shape)))
    if Y.shape[0] != design_matrix.shape[0]:
        raise ValueError(
            'the number %d of rows in Effects and in design matrix (%d)'
            'do not match' % (Y.shape[0], design_matrix.shape[0]))
    
    # here we manipulate design matrix and contrast to obatin X and X0
    X = design_matrix.values
    X_null = X - np.dot(np.dot(X, contrast), np.linalg.pinv(contrast))
    # get parameters estimates for the full model
    beta_, V2, log_likelihood = mixed_model_inference(X, Y, V1, n_iter=n_iter)
    # parameters estimates for the reduced model
    _, _, log_likelihood_null = mixed_model_inference(
        X_null, Y, V1, n_iter=n_iter)
    # compute the log of likrelihood ratio between the two models
    logl_ratio = np.maximum(log_likelihood - log_likelihood_null, 0)
    # convert the permuted log-likelihood to z-scores
    z_ = np.sqrt(2 * logl_ratio) * np.sign(beta_)
            
    max_diff_loglike = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_max_llr)(X, contrast, Y, V1, n_iter)
        for _ in range(n_perm))

    # convert the permuted log-likelihood to z-scores
    max_diff_z = np.sqrt(2 * np.array(sorted(max_diff_loglike)))

    # Generate outputs
    beta = masker.inverse_transform(beta_)
    z_map = masker.inverse_transform(z_)
    second_level_variance = masker.inverse_transform(V2)
    return (beta, second_level_variance, z_map, max_diff_z)


def non_parametric_mixed_effects_inference(
        second_level_input, confounds=None, design_matrix=None,
        second_level_contrast=None, mask=None, smoothing_fwhm=None,
        model_intercept=True, n_perm=10000, two_sided_test=False,
        random_state=None, n_jobs=1, verbose=0):
    """Generate p-values corresponding to the contrasts provided
    based on permutation testing. This fuction reuses the 'permuted_ols'
    function Nilearn.

    Parameters
    ----------
    second_level_input: pandas DataFrame or list of Niimg-like objects.

        If a pandas DataFrame, then they have to contain subject_label,
        map_name and effects_map_path. It can contain multiple maps that
        would be selected during contrast estimation with the argument
        first_level_contrast of the compute_contrast function. The
        DataFrame will be sorted based on the subject_label column to avoid
        order inconsistencies when extracting the maps. So the rows of the
        automatically computed design matrix, if not provided, will
        correspond to the sorted subject_label column.

        If list of Niimg-like objects then this is taken literally as Y
        for the model fit and design_matrix must be provided.

    confounds: pandas DataFrame, optional
        Must contain a subject_label column. All other columns are
        considered as confounds and included in the model. If
        design_matrix is provided then this argument is ignored.
        The resulting second level design matrix uses the same column
        names as in the given DataFrame for confounds. At least two columns
        are expected, "subject_label" and at least one confound.

    design_matrix: pandas DataFrame, optional
        Design matrix to fit the GLM. The number of rows
        in the design matrix must agree with the number of maps derived
        from second_level_input.
        Ensure that the order of maps given by a second_level_input
        list of Niimgs matches the order of the rows in the design matrix.

    second_level_contrast: str or array of shape (n_col), optional
        Where ``n_col`` is the number of columns of the design matrix.
        The default (None) is accepted if the design matrix has a single
        column, in which case the only possible contrast array([1]) is
        applied; when the design matrix has multiple columns, an error is
        raised.

    mask: Niimg-like, NiftiMasker or MultiNiftiMasker object, optional,
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters. Automatic mask computation assumes first level imgs have
        already been masked.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    model_intercept : bool,
      If True, a constant column is added to the confounding variates
      unless the tested variate is already the intercept.

    n_perm : int,
      Number of permutations to perform.
      Permutations are costly but the more are performed, the more precision
      one gets in the p-values estimation.

    two_sided_test : boolean,
      If True, performs an unsigned t-test. Both positive and negative
      effects are considered; the null hypothesis is that the effect is zero.
      If False, only positive effects are considered as relevant. The null
      hypothesis is that the effect is zero or negative.

    random_state : int or None,
      Seed for random number generator, to have the same permutations
      in each computing units.

    n_jobs : int,
      Number of parallel workers.
      If -1 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (abs(n_jobs) - 1)
      ones will be used.

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    neg_log_corrected_pvals_img: Nifti1Image
        The image which contains negative logarithm of the
        corrected p-values
    """
    _check_second_level_input(second_level_input, design_matrix,
                              flm_object=False, df_object=False)
    _check_confounds(confounds)
    _check_design_matrix(design_matrix)

    # Report progress
    t0 = time.time()
    if verbose > 0:
        sys.stderr.write("Fitting second level model...")

    # Select sample map for masker fit and get subjects_label for design
    sample_map = mean_img(second_level_input)

    # Learn the mask. Assume the first level imgs have been masked.
    if not isinstance(mask, NiftiMasker):
        masker = NiftiMasker(
            mask_img=mask, smoothing_fwhm=smoothing_fwhm,
            memory=Memory(None), verbose=max(0, verbose - 1),
            memory_level=1)
    else:
        masker = clone(mask)
        if smoothing_fwhm is not None:
            if getattr(masker, 'smoothing_fwhm') is not None:
                warn('Parameter smoothing_fwhm of the masker overriden')
                setattr(masker, 'smoothing_fwhm', smoothing_fwhm)
    masker.fit(sample_map)

    # Report progress
    if verbose > 0:
        sys.stderr.write("\nComputation of second level model done in "
                         "%i seconds\n" % (time.time() - t0))

    # Check and obtain the contrast
    contrast = _get_contrast(second_level_contrast, design_matrix)

    # Get effect_maps
    effect_maps = _infer_effect_maps(second_level_input, None)

    # Check design matrix and effect maps agree on number of rows
    _check_effect_maps(effect_maps, design_matrix)

    # Obtain tested_var
    if contrast in design_matrix.columns.tolist():
        tested_var = np.asarray(design_matrix[contrast])

    # Mask data
    target_vars = masker.transform(effect_maps)

    # Perform massively univariate analysis with permuted OLS
    neg_log_pvals_permuted_ols, _, _ = permuted_ols(
        tested_var, target_vars, model_intercept=model_intercept,
        n_perm=n_perm, two_sided_test=two_sided_test,
        random_state=random_state, n_jobs=n_jobs, verbose=max(0, verbose - 1))
    neg_log_corrected_pvals_img = masker.inverse_transform(
        np.ravel(neg_log_pvals_permuted_ols))

    return neg_log_corrected_pvals_img
