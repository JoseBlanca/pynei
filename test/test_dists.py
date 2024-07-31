import math

import numpy
import pandas

from pynei.dists import Distances, _KosmanDistCalculator
from pynei import Variants, calc_pairwise_kosman_dists


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
    dists = calc_pairwise_kosman_dists(vars, num_processes=1)
    assert numpy.allclose(dists.dist_vector, expected)
    return

    dists_emb = calc_kosman_pairwise_dists(gts, use_approx_embedding_algorithm=True)
    dists = dists.square_dists
    dists_emb = dists_emb.square_dists
    dists_emb = dists_emb.loc[dists.index, :].loc[:, dists.index]
    assert numpy.corrcoef(dists_emb.values.flat, dists.values.flat)[0, 1] > 0.99
