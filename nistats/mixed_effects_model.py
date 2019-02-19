"""
Extension of second_level_model to handle mixed effects, ie two levels of variance
corresponding to intra- and inter-subject variability respectively
"""
import numpy as np
from second_level_model import SecondLevelModel


def _mixed_log_likelihood(data, mean, V1, V2):
    """ return the log-likelighood of the data under the composite variance model
    """
    total_variance = V1 + V2
    logl = np.sum(((data - mean) ** 2) / total_variance, 0)
    logl += np.sum(np.log(total_variance), 0)
    logl += np.log(2 * np.pi) * X.shape[0]
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

    logl = _mixed_log_likelihood(Y, Y_, V1, V2_)
    if self.verbose:
        print('Average log-likelihood: ', logl_.mean())

    for i in range(n_iter):
        beta_, Y_, V2_ = _em_step(Y, Y_, V1, V2, X, pinv_X)

        logl_ = _mixed_log_likelihood(Y, Y_, V1, V2_)
        if self.verbose:
            if (logl_ < (logl - 100 * np.finfo(float).eps)).any():
                raise ValueError('The log-likelihood cannot decrease')
            logl = logl_
            print('Iteration %d, average log-likelihood: %f' % (
                    i, logl_.mean()))
    return beta_, V2_, logl_


def compute_contrast(
    masker, effects_maps, variance_maps, design_matrix, contrast,
    output_type='z_score', n_perm=1000, n_iter=5):
    """ Compute a second-level contrast given brain maps

    Parameters
    ----------
    masker: NiiftiMasker
    effects_maps: niimg
    variance_maps: niimg
    design_matrix: numpy array

    Returns
    -------
    effect size map
    second_level variance map
    log-likelihood map
    """
    Y = masker.transform(effect_maps)
    V1 = masker.transform(variance_maps)

    # here should manipulate design matrix and contrast to obatin X and X0
    X = design_matrix.values
    X_null = X - np.dot(np.dot(X, contrast), np.linalg.pinv(contrast))
    beta, V2, log_likelihood = mixed_model_inference(X, Y, V1, n_iter=n_iter)
    _, _, log_likelihood_null = mixed_model_inference(
        X_null, Y, V1, n_iter=n_iter)
    log_likelihood_ratio = np.maximum(log_likelihood - log_likelihood_null, 0)

    max_diff_loglike = []
    for _ in range(n_perm):
        X_ = np.random.shuffle(X)
        X_null_ = X_ - np.dot(np.dot(X_, contrast), np.linalg.pinv(contrast))
        _, _, log_likelihood_ = mixed_model_inference(X_, Y, V1, n_iter=n_iter)
        _, _, log_likelihood_null_ = mixed_model_inference(
            X_null_, Y, V1, n_iter=n_iter)
        log_likelihood_ratio_ = np.maximum(
            log_likelihood_ - log_likelihood_null_, 0)
        max_diff_loglike.append(log_likelihood_ratio_.max())

    beta_map = masker.inverse_transform(beta)
    second_level_variance_map = masker.inverse_transform(V2)
    log_likelihood_ratio_map = masker.inverse_transform(log_likelihood_ratio)
    return beta_map, second_level_variance_map, log_likelihood_ratio_map


def two_sample_ftest(Y, V1, group, n_iter=5, verbose=False):
    """Returns the mixed effects t-stat for each row of the X
    (one sample test)
    This uses the Formula in Roche et al., NeuroImage 2007

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the data
    V1: array of shape (n_samples, n_tests)
        first-level variance assocated with the data
    group: array of shape (n_samples)
       a vector of indicators yielding the samples membership
    n_iter: int, optional,
           number of iterations of the EM algorithm
    verbose: bool, optional, verbosity mode

    Returns
    -------
    tstat: array of shape (n_tests),
           statistical values obtained from the likelihood ratio test
    """
    # check that group is correct
    if group.size != Y.shape[0]:
        raise ValueError('The number of labels is not the number of samples')
    if (np.unique(group) != np.array([0, 1])).all():
        raise ValueError('group should be composed only of zeros and ones')

    # create design matrices
    X = np.vstack((np.ones_like(group), group)).T
    return mfx_stat(Y, V1, X, 1, n_iter=n_iter, verbose=verbose,
                    return_t=False, return_f=True)[0]


def two_sample_ttest(Y, V1, group, n_iter=5, verbose=False):
    """Returns the mixed effects t-stat for each row of the X
    (one sample test)
    This uses the Formula in Roche et al., NeuroImage 2007

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the data
    V1: array of shape (n_samples, n_tests)
        first-level variance assocated with the data
    group: array of shape (n_samples)
       a vector of indicators yielding the samples membership
    n_iter: int, optional,
           number of iterations of the EM algorithm
    verbose: bool, optional, verbosity mode

    Returns
    -------
    tstat: array of shape (n_tests),
           statistical values obtained from the likelihood ratio test
    """
    X = np.vstack((np.ones_like(group), group)).T
    return mfx_stat(Y, V1, X, 1, n_iter=n_iter, verbose=verbose,
                    return_t=True)[0]


def one_sample_ftest(Y, V1, n_iter=5, verbose=False):
    """Returns the mixed effects F-stat for each row of the X
    (one sample test)
    This uses the Formula in Roche et al., NeuroImage 2007

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the data
    V1: array of shape (n_samples, n_tests)
        first-level variance ssociated with the data
    n_iter: int, optional,
           number of iterations of the EM algorithm
    verbose: bool, optional, verbosity mode

    Returns
    -------
    fstat, array of shape (n_tests),
           statistical values obtained from the likelihood ratio test
    sign, array of shape (n_tests),
          sign of the mean for each test (allow for post-hoc signed tests)
    """
    return mfx_stat(Y, V1, np.ones((Y.shape[0], 1)), 0, n_iter=n_iter,
                    verbose=verbose, return_t=False, return_f=True)[0]


def one_sample_ttest(Y, V1, n_iter=5, verbose=False):
    """Returns the mixed effects t-stat for each row of the X
    (one sample test)
    This uses the Formula in Roche et al., NeuroImage 2007

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the observations
    V1: array of shape (n_samples, n_tests)
        first-level variance associated with the observations
    n_iter: int, optional,
           number of iterations of the EM algorithm
    verbose: bool, optional, verbosity mode

    Returns
    -------
    tstat: array of shape (n_tests),
           statistical values obtained from the likelihood ratio test
    """
    return mfx_stat(Y, V1, np.ones((Y.shape[0], 1)), 0, n_iter=n_iter,
                    verbose=verbose, return_t=True)[0]


def mfx_stat(Y, V1, X, column, n_iter=5, return_t=True,
             return_f=False, return_effect=False,
             return_var=False, verbose=False):
    """Run a mixed-effects model test on the column of the design matrix

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the data
    V1: array of shape (n_samples, n_tests)
        first-level variance assocated with the data
    X: array of shape(n_samples, n_regressors)
       the design matrix of the model
    column: int,
            index of the column of X to be tested
    n_iter: int, optional,
           number of iterations of the EM algorithm
    return_t: bool, optional,
              should one return the t test (True by default)
    return_f: bool, optional,
              should one return the F test (False by default)
    return_effect: bool, optional,
              should one return the effect estimate (False by default)
    return_var: bool, optional,
              should one return the variance estimate (False by default)

    verbose: bool, optional, verbosity mode

    Returns
    -------
    (tstat, fstat, effect, var): tuple of arrays of shape (n_tests),
                                 those required by the input return booleans

    """
    # check that X/columns are correct
    column = int(column)
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X.shape[0] is not the number of samples')
    if (column > X.shape[1]):
        raise ValueError('the column index is more than the number of columns')

    # create design matrices
    contrast_mask = 1 - np.eye(X.shape[1])[column]
    X0 = X * contrast_mask

    # instantiate the mixed effects models
    model_0 = MixedEffectsModel(X0, n_iter=n_iter, verbose=verbose).fit(Y, V1)
    model_1 = MixedEffectsModel(X, n_iter=n_iter, verbose=verbose).fit(Y, V1)

    # compute the log-likelihood ratio statistic
    fstat = 2 * (model_1.log_like(Y, V1) - model_0.log_like(Y, V1))
    fstat = np.maximum(0, fstat)
    sign = np.sign(model_1.beta_[column])

    output = ()
    if return_t:
        output += (np.sqrt(fstat) * sign,)
    if return_f:
        output += (fstat,)
    if return_var:
        output += (model_1.V2,)
    if return_effect:
        output += (model_1.beta_[column],)
    return output
