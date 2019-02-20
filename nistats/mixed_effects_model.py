"""
Extension of second_level_model to handle mixed effects,
ie two levels of variance corresponding to
intra- and inter-subject variability respectivelyself.

Author: Bertrand Thirion
"""
import numpy as np
from nistats.second_level_model import SecondLevelModel
from joblib import Parallel, delayed

def _mixed_log_likelihood(data, mean, V1, V2):
    """ return the log-likelighood of the data under the composite variance model
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

    logl = _mixed_log_likelihood(Y, Y_, V1, V2_)
    if verbose:
        print('Average log-likelihood: ', logl.mean())

    for i in range(n_iter):
        beta_, Y_, V2_ = _em_step(Y, Y_, V1, V2_, X, pinv_X)
        logl_ = _mixed_log_likelihood(Y, Y_, V1, V2_)
        if verbose:
            if (logl_ < (logl - 100 * np.finfo(float).eps)).any():
                raise ValueError('The log-likelihood cannot decrease')
            logl = logl_
            print('Iteration %d, average log-likelihood: %f' % (
                    i, logl_.mean()))
    return beta_, V2_, logl_


def _randomize_design(X):
    """small utility to randomize the design matrix"""
    X_ = X.copy()
    np.random.shuffle(X_)
    if (X_ == X).all():
        sign_swap = 2 * (np.random.rand(*X.shape) > .5) - 1
        X_ *= sign_swap
    return X_

    
def mixed_effects_likelihood_ratio_test(
    masker, effects, variance, design_matrix, contrast,
    n_perm=10, n_iter=5, n_jobs=1):
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
    Y = masker.transform(effects)
    V1 = masker.transform(variance)

    # here we manipulate design matrix and contrast to obatin X and X0
    X = design_matrix.values
    X_null = X - np.dot(np.dot(X, contrast), np.linalg.pinv(contrast))
    beta_, V2, log_likelihood = mixed_model_inference(X, Y, V1, n_iter=n_iter)
    _, _, log_likelihood_null = mixed_model_inference(
        X_null, Y, V1, n_iter=n_iter)
    logl_ratio = np.maximum(log_likelihood - log_likelihood_null, 0)

    def permuted_max(X, contrast, Y, V1, n_iter):
        X_ = _randomize_design(X)
        X_null_ = X - np.dot(np.dot(X_, contrast), np.linalg.pinv(contrast))
        _, _, log_likelihood_ = mixed_model_inference(X_, Y, V1, n_iter=n_iter)
        _, _, log_likelihood_null_ = mixed_model_inference(
            X_null_, Y, V1, n_iter=n_iter)
        log_likelihood_ratio_ = np.maximum(
            log_likelihood_ - log_likelihood_null_, 0)
        return log_likelihood_ratio_.max()
        
    max_diff_loglike = Parallel(n_jobs=n_jobs)(
        delayed(permuted_max)(X, contrast, Y, V1, n_iter)
        for _ in range(n_perm))
    
    max_diff_loglike = np.array(sorted(max_diff_loglike))
    beta = masker.inverse_transform(beta_)
    second_level_variance = masker.inverse_transform(V2)
    log_likelihood_ratio = masker.inverse_transform(logl_ratio)
    return (beta, second_level_variance, log_likelihood_ratio,
            max_diff_loglike)

