import math

import numpy
import pandas

from pynei.dists import (
    Distances,
    _KosmanDistCalculator,
    calc_euclidean_pairwise_dists,
    calc_jost_dest_pop_dists,
    _DestDistCalculator,
)
from pynei import Variants, calc_pairwise_kosman_dists
from pynei.dists import _calc_pops_idxs


def test_distances():
    dists_orig = [[0, 0.5, 0.75], [0.5, 0, 0.3], [0.75, 0.3, 0]]
    dists = Distances.from_square_dists(pandas.DataFrame(dists_orig))
    assert numpy.allclose(dists.dist_vector, [0.5, 0.75, 0.3])

    dists = Distances([0.5, 0.75, 0.3])
    assert numpy.allclose(dists.dist_vector, [0.5, 0.75, 0.3])

    assert numpy.allclose(dists.square_dists.values, dists_orig)


def test_kosman_2_indis():
    a = numpy.array(
        [
            [-1, -1],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    b = numpy.array(
        [
            [1, 1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    gt_array = numpy.stack((a, b), axis=1)
    chunk = next(Variants.from_gt_array(gt_array).iter_vars_chunks())
    dist, n_snps = _KosmanDistCalculator(chunk).calc_dist_sum_and_n_snps_btw_two_indis(
        0, 1
    )
    assert math.isclose(dist, 3.0)
    assert n_snps == 9
    distance = _KosmanDistCalculator(chunk).calc_dist_btw_two_indis(0, 1)
    assert math.isclose(distance, 1 / 3)

    c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    gt_array = numpy.stack((c, d), axis=1)
    chunk = next(Variants.from_gt_array(gt_array).iter_vars_chunks())
    dist, n_snps = _KosmanDistCalculator(chunk).calc_dist_sum_and_n_snps_btw_two_indis(
        0, 1
    )
    assert math.isclose(dist, 0.0)
    assert n_snps == c.shape[0]
    distance = _KosmanDistCalculator(chunk).calc_dist_btw_two_indis(0, 1)
    assert math.isclose(distance, 0.0)

    gt_array = numpy.stack((b, d), axis=1)
    chunk = next(Variants.from_gt_array(gt_array).iter_vars_chunks())
    distance = _KosmanDistCalculator(chunk).calc_dist_btw_two_indis(0, 1)
    assert math.isclose(distance, 0.45)


def test_kosman_missing():
    a = numpy.array(
        [
            [-1, -1],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    b = numpy.array(
        [
            [1, 1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    gt_array = numpy.stack((a, b), axis=1)
    chunk = next(Variants.from_gt_array(gt_array).iter_vars_chunks())
    distance_ab = _KosmanDistCalculator(chunk).calc_dist_btw_two_indis(0, 1)

    a = numpy.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    b = numpy.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    gt_array = numpy.stack((a, b), axis=1)
    chunk = next(Variants.from_gt_array(gt_array).iter_vars_chunks())
    distance_cd = _KosmanDistCalculator(chunk).calc_dist_btw_two_indis(0, 1)

    assert distance_ab == distance_cd


def test_kosman_pairwise():
    a = numpy.array(
        [
            [-1, -1],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    b = numpy.array(
        [
            [1, 1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 2],
        ]
    )
    c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    gts = numpy.stack((a, b, c, d), axis=0)
    gts = numpy.transpose(gts, axes=(1, 0, 2)).astype(numpy.int16)
    vars = Variants.from_gt_array(gts, samples=["a", "b", "c", "d"])

    expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.0]
    dists = calc_pairwise_kosman_dists(vars)
    assert numpy.allclose(dists.dist_vector, expected)

    dists_emb = calc_pairwise_kosman_dists(vars, use_approx_embedding_algorithm=True)
    dists = dists.square_dists
    dists_emb = dists_emb.square_dists
    dists_emb = dists_emb.loc[dists.index, :].loc[:, dists.index]
    assert numpy.corrcoef(dists_emb.values.flat, dists.values.flat)[0, 1] > 0.99

    dists_emb = calc_pairwise_kosman_dists(
        vars, use_approx_embedding_algorithm=True, num_processes=2
    )
    dists_emb = dists_emb.square_dists
    dists_emb = dists_emb.loc[dists.index, :].loc[:, dists.index]
    assert numpy.corrcoef(dists_emb.values.flat, dists.values.flat)[0, 1] > 0.99


def test_euclidean_dists():
    num_samples = 4
    num_traits = 10
    numpy.random.seed(42)
    samples = pandas.DataFrame(numpy.random.uniform(size=(num_samples, num_traits)))
    dists = calc_euclidean_pairwise_dists(samples)
    expected = [0.8160523, 1.4245896, 1.74402628, 1.37436733, 1.84068677, 1.00002389]
    assert numpy.allclose(dists.dist_vector, expected)


def test_dest_jost_distance():
    gts = [
        [  #          sample pop is_het tot_het freq_het
            (1, 1),  #    1     1
            (1, 3),  #    2     1     1
            (1, 2),  #    3     1     1
            (1, 4),  #    4     1     1
            (3, 3),  #    5     1             3     3/5=0.6
            (3, 2),  #    6     2     1
            (3, 4),  #    7     2     1
            (2, 2),  #    8     2
            (2, 4),  #    9     2     1
            (4, 4),  #   10     2
            (-1, -1),  # 11     2             3     3/6=0.5
        ],
        [
            (1, 3),
            (1, 1),
            (1, 1),
            (1, 3),
            (3, 3),
            (3, 2),
            (3, 4),
            (2, 2),
            (2, 4),
            (4, 4),
            (-1, -1),
        ],
    ]
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    snps = Variants.from_gt_array(numpy.array(gts), samples=samples)

    pop1 = [1, 2, 3, 4, 5]
    pop2 = [6, 7, 8, 9, 10, 11]
    pops = {"pop1": pop1, "pop2": pop2}

    calc_dists = _DestDistCalculator(
        pop_idxs=_calc_pops_idxs({"pop1": pop1, "pop2": pop2}, snps.samples),
        sorted_pop_ids=["pop1", "pop2"],
        min_num_genotypes=1,
        ploidy=snps.ploidy,
        alleles=None,
    )
    chunk = next(snps.iter_vars_chunks())
    dists = calc_dists(chunk)
    expected = numpy.array([0.49090909, 0.77931034])
    assert numpy.allclose(dists, expected)

    dist = calc_jost_dest_pop_dists(snps, pops=pops, min_num_samples=0)
    assert numpy.allclose(dist.dist_vector, [0.65490196])

    dists = calc_jost_dest_pop_dists(snps, pops=pops, min_num_samples=6)
    assert numpy.all(numpy.isnan(dists.dist_vector))
