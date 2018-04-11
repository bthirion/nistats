from __future__ import with_statement

import numpy as np

from nistats.first_level_model import run_glm
from nistats.contrasts import compute_contrast, _fixed_effect_contrast

from numpy.testing import assert_almost_equal
from scipy.linalg import sqrtm


def test_Tcontrast():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    labels, results = run_glm(Y, X, 'ar1')
    con_val = np.eye(q)[0]
    z_vals = compute_contrast(labels, results, con_val).z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    for model in ['ols', 'ar1']:
        labels, results = run_glm(Y, X, model)
        for con_val in [np.eye(q)[0], np.eye(q)[:3]]:
            z_vals = compute_contrast(
                labels, results, con_val, contrast_type='F').z_score()
            assert_almost_equal(z_vals.mean(), 0, 0)
            assert_almost_equal(z_vals.std(), 1, 0)


def test_t_contrast_add():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = run_glm(Y, X, 'ols')
    c1, c2 = np.eye(q)[0], np.eye(q)[1]
    con = compute_contrast(lab, res, c1) + compute_contrast(lab, res, c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_fixed_effect_contrast():
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = run_glm(Y, X, 'ols')
    c1, c2 = np.eye(q)[0], np.eye(q)[1]
    con = _fixed_effect_contrast([lab, lab], [res, res], [c1, c2])
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_F_contrast_add():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = run_glm(Y, X, 'ar1')
    c1, c2 = np.eye(q)[:2], np.eye(q)[2:4]
    con = compute_contrast(lab, res, c1) + compute_contrast(lab, res, c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

    # first test with dependent contrast
    con1 = compute_contrast(lab, res, c1)
    con2 = compute_contrast(lab, res, c1) + compute_contrast(lab, res, c1)
    assert_almost_equal(con1.effect * 2, con2.effect)
    assert_almost_equal(con1.variance * 2, con2.variance)
    assert_almost_equal(con1.stat() * 2, con2.stat())


def test_contrast_mul():
    # new API
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = run_glm(Y, X, 'ar1')
    for c1 in [np.eye(q)[0], np.eye(q)[:3]]:
        con1 = compute_contrast(lab, res, c1)
        con2 = con1 * 2
        assert_almost_equal(con1.effect * 2, con2.effect)
        # assert_almost_equal(con1.variance * 2, con2.variance) FIXME
        # assert_almost_equal(con1.stat() * 2, con2.stat()) FIXME
        assert_almost_equal(con1.z_score(), con2.z_score())


def test_contrast_values():
    # new API
    # but this test is circular and should be removed
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    lab, res = run_glm(Y, X, 'ar1', bins=1)
    # t test
    cval = np.eye(q)[0]
    con = compute_contrast(lab, res, cval)
    summary_stats = res.values()[0]
    t_ref = cval.dot(summary_stats.theta) /\
        np.ravel(np.sqrt(summary_stats.vcov(matrix=np.atleast_2d(cval))))
    assert_almost_equal(np.ravel(con.stat()), t_ref)
    # F test
    cval = np.eye(q)[:3]
    con = compute_contrast(lab, res, cval)
    cbeta = np.atleast_2d(np.dot(cval, summary_stats.theta))
    invcov = np.linalg.inv(
        summary_stats.vcov(matrix=cval, dispersion=1.))
    wcbeta = np.dot(sqrtm(invcov), cbeta)
    F_ref = np.sum(wcbeta ** 2, 0) / 3 / summary_stats.dispersion
    # F_ref = list(res.values())[0].Fcontrast(cval).F
    # Note that the values are not strictly equal,
    # this seems to be related to a bug in Mahalanobis
    assert_almost_equal(np.ravel(con.stat()), F_ref, 3)
