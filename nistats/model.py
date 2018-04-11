# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module implement classes to handle statistical tests on likelihood models

Author: Bertrand Thirion, 2011--2015
"""

import numpy as np
from scipy.linalg import inv
from scipy.stats import t as t_distribution

from nibabel.onetime import setattr_on_read

from .utils import positive_reciprocal

# Inverse t cumulative distribution
inv_t_cdf = t_distribution.ppf


class LikelihoodModelResults(object):
    ''' Class to contain results from likelihood models '''

    # This is the class in which things like AIC, BIC, llf can be implemented as
    # methods, not computed in, say, the fit method of OLSModel

    def __init__(self, theta, Y, model, cov=None, dispersion=1., nuisance=None,
                 rank=None):
        ''' Set up results structure

        Parameters
        ----------
        theta : ndarray
            parameter estimates from estimated model

        Y : ndarray
            data

        model : ``LikelihoodModel`` instance
            model used to generate fit

        cov : None or ndarray, optional
            covariance of thetas

        dispersion : scalar, optional
            multiplicative factor in front of `cov`

        nuisance : None of ndarray
            parameter estimates needed to compute logL

        rank : None or scalar
            rank of the model.  If rank is not None, it is used for df_model
            instead of the usual counting of parameters.

        Notes
        -----
        The covariance of thetas is given by:

            dispersion * cov

        For (some subset of models) `dispersion` will typically be the mean
        square error from the estimated model (sigma^2)
        '''
        self.theta = theta
        self.Y = Y
        self.model = model
        if cov is None:
            self.cov = self.model.information(self.theta,
                                              nuisance=self.nuisance)
        else:
            self.cov = cov
        self.dispersion = dispersion
        self.nuisance = nuisance

        self.df_total = Y.shape[0]
        self.df_model = model.df_model
        # put this as a parameter of LikelihoodModel
        self.df_resid = self.df_total - self.df_model
        
    @setattr_on_read
    def logL(self):
        """
        The maximized log-likelihood
        """
        return self.model.logL(self.theta, self.Y, nuisance=self.nuisance)

    def t(self, column=None):
        """
        Return the (Wald) t-statistic for a given parameter estimate.

        Use Tcontrast for more complicated (Wald) t-statistics.
        """

        if column is None:
            column = range(self.theta.shape[0])

        column = np.asarray(column)
        _theta = self.theta[column]
        _cov = self.vcov(column=column)
        if _cov.ndim == 2:
            _cov = np.diag(_cov)
        _t = _theta * positive_reciprocal(np.sqrt(_cov))
        return _t

    def vcov(self, matrix=None, column=None, dispersion=None, other=None):
        """ Variance/covariance matrix of linear contrast

        Parameters
        ----------
        matrix: (dim, self.theta.shape[0]) array, optional
            numerical contrast specification, where ``dim`` refers to the
            'dimension' of the contrast i.e. 1 for t contrasts, 1 or more
            for F contrasts.

        column: int, optional
            alternative way of specifying contrasts (column index)

        dispersion: float or (n_voxels,) array, optional
            value(s) for the dispersion parameters

        other: (dim, self.theta.shape[0]) array, optional
            alternative contrast specification (?)

        Returns
        -------
        cov: (dim, dim) or (n_voxels, dim, dim) array
            the estimated covariance matrix/matrices

        Returns the variance/covariance matrix of a linear contrast of the
        estimates of theta, multiplied by `dispersion` which will often be an
        estimate of `dispersion`, like, sigma^2.

        The covariance of interest is either specified as a (set of) column(s)
        or a matrix.
        """
        if self.cov is None:
            raise ValueError('need covariance of parameters for computing' +\
                             '(unnormalized) covariances')

        if dispersion is None:
            dispersion = self.dispersion

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return self.cov[column, column] * dispersion
            else:
                return self.cov[column][:, column] * dispersion

        elif matrix is not None:
            if other is None:
                other = matrix
            tmp = np.dot(matrix, np.dot(self.cov, np.transpose(other)))
            if np.isscalar(dispersion):
                return tmp * dispersion
            else:
                return tmp[:, :, np.newaxis] * dispersion
        if matrix is None and column is None:
            return self.cov * dispersion
