import random
import math
from functools import partial

import numpy
import pytest

from pynei.variants import Genotypes, Variants
from pynei.ld import _calc_rogers_huff_r2, calc_pairwise_rogers_huff_r2
from .var_generator import generate_vars


def create_gts_for_sample(num_samples, geno_freqs):
    gts = sorted(geno_freqs.keys())
    weights = [geno_freqs[gt] for gt in gts]
    gts = random.choices(gts, weights=weights, k=num_samples)
    return gts


def create_gts(num_vars, num_samples, independence_rate, geno_freqs):
    ref_snp_gts = create_gts_for_sample(num_samples, geno_freqs)
    rng = numpy.random.default_rng()
    column_idxs = numpy.arange(num_samples)
    gts = []
    for _ in range(num_vars):
        independent_snp_gts = create_gts_for_sample(num_samples, geno_freqs)
        snp_idxs = rng.choice(
            [0, 1], size=num_samples, p=[1 - independence_rate, independence_rate]
        )
        snp_gts = numpy.array([ref_snp_gts, independent_snp_gts])[
            snp_idxs, column_idxs, :
        ]
        gts.append(snp_gts)
    gts = Genotypes(numpy.array(gts))
    return gts


def _bivmom(vec0, vec1):
    """
    Calculate means, variances, the covariance, from two data vectors.
    On entry, vec0 and vec1 should be vectors of numeric values and
    should have the same length.  Function returns m0, v0, m1, v1,
    cov, where m0 and m1 are the means of vec0 and vec1, v0 and v1 are
    the variances, and cov is the covariance.
    """
    m0 = m1 = v0 = v1 = cov = 0
    for x, y in zip(vec0, vec1):
        m0 += x
        m1 += y
        v0 += x * x
        v1 += y * y
        cov += x * y
    n = len(vec0)
    assert n == len(vec1)
    n = float(n)
    m0 /= n
    m1 /= n
    v0 /= n
    v1 /= n
    cov /= n

    cov -= m0 * m1
    v0 -= m0 * m0
    v1 -= m1 * m1

    return m0, v0, m1, v1, cov


def _get_r(Y, Z):
    """
    Estimates r w/o info on gametic phase.  Also works with gametic
    data, in which case Y and Z should be vectors of 0/1 indicator
    variables.
    Uses the method of Rogers and Huff 2008.
    """
    mY, vY, mZ, vZ, cov = _bivmom(Y, Z)
    if False:
        print("cov", cov)
        print("vY", vY)
        print("vZ", vZ)
    return cov / math.sqrt(vY * vZ)


def test_ld_calc():
    numpy.random.seed(42)

    num_samples = 10000
    num_vars = 10
    geno_freqs = {(0, 0): 0.8, (1, 1): 0.2, (-1, -1): 0.01}

    for independence_rate in [0, 1, 0.5]:
        gts = create_gts(
            num_vars,
            num_samples,
            independence_rate=independence_rate,
            geno_freqs=geno_freqs,
        )
        gts = gts.to_012()
        r2 = _calc_rogers_huff_r2(gts, gts, debug=False)

        non_diag_mask = ~numpy.eye(r2.shape[0], dtype=bool)
        non_diag_r2 = r2[non_diag_mask]
        non_diag_r = numpy.sqrt(numpy.abs(non_diag_r2))
        assert numpy.allclose(non_diag_r, 1 - independence_rate, atol=0.2)
        assert numpy.allclose(r2[~non_diag_mask], 1.0, atol=0.1)

    Y = [
        2,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        2,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
    ]

    Z = [
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        0,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        1,
        2,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        2,
        1,
        2,
        2,
        2,
        1,
        1,
    ]
    yz_r = _get_r(Y, Z)
    yy_r = _get_r(Y, Y)
    zz_r = _get_r(Z, Z)
    gts = numpy.array([Y, Z, Z])
    r = _calc_rogers_huff_r2(gts, gts)
    assert math.isclose(r[0, 1], yz_r, abs_tol=1e-4)
    assert math.isclose(r[0, 0], yy_r, abs_tol=1e-4)
    assert math.isclose(r[1, 1], zz_r, abs_tol=1e-4)

    gts = [0] * 100 + [1]
    gts = numpy.array([gts])
    with pytest.raises(ValueError):
        _calc_rogers_huff_r2(gts, gts)
    _calc_rogers_huff_r2(gts, gts, check_no_mafs_above=None)


def test_pairwiseld():
    gts = [
        [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 1), (0, 0)],
        [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0)],
        [(0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (1, 1), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 1), (0, 1), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    ]
    vars = Variants.from_gt_array(gts)
    r2s = [ld_res.r2 for ld_res in calc_pairwise_rogers_huff_r2(vars)]
    chunk = next(vars.iter_vars_chunks())
    r2_array = _calc_rogers_huff_r2(
        chunk.gts.to_012(),
        chunk.gts.to_012(),
    )
    assert numpy.allclose(r2s, r2_array[numpy.triu_indices(r2_array.shape[0], k=1)])

    vars = Variants.from_gt_array(gts)
    vars.desired_num_vars_per_chunk = 2
    r2s2 = [ld_res.r2 for ld_res in calc_pairwise_rogers_huff_r2(vars)]
    assert numpy.allclose(sorted(r2s), sorted(r2s2))

    create_gts_funct = partial(
        create_gts, independence_rate=0.5, geno_freqs={(0, 0): 0.5, (1, 1): 0.5}
    )
    vars = generate_vars(
        num_chroms=2,
        num_vars_per_chrom=10,
        dist_between_vars=1000,
        create_gts_funct=create_gts_funct,
        num_samples=10,
        chunk_size=3,
    )
    calc_pairwise_rogers_huff_r2(vars)
    # calc_pairwise_rogers_huff_r2(vars, max_dist=2001)
