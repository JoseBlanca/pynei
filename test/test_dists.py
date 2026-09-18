import itertools
import math
import random

import numpy

from .var_generators import create_sample_names
import pandas

from pynei.dists import (
    Distances,
    _KosmanDistCalculator,
    calc_pairwise_euclidean_dists,
    calc_jost_dest_pop_dists,
    _DestDistCalculator,
)
from pynei import Variants, calc_pairwise_kosman_dists, filter_by_missing_data
from pynei.dists import _calc_pops_idxs, _calc_kosman_dist_for_chunk


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
    chunk = next(
        Variants.from_gt_array(
            gt_array, samples=create_sample_names(gt_array)
        ).iter_vars_chunks()
    )
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
    chunk = next(
        Variants.from_gt_array(
            gt_array, samples=create_sample_names(gt_array)
        ).iter_vars_chunks()
    )
    dist, n_snps = _KosmanDistCalculator(chunk).calc_dist_sum_and_n_snps_btw_two_indis(
        0, 1
    )
    assert math.isclose(dist, 0.0)
    assert n_snps == c.shape[0]
    distance = _KosmanDistCalculator(chunk).calc_dist_btw_two_indis(0, 1)
    assert math.isclose(distance, 0.0)

    gt_array = numpy.stack((b, d), axis=1)
    chunk = next(
        Variants.from_gt_array(
            gt_array, samples=create_sample_names(gt_array)
        ).iter_vars_chunks()
    )
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
    chunk = next(
        Variants.from_gt_array(
            gt_array, samples=create_sample_names(gt_array)
        ).iter_vars_chunks()
    )
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
    chunk = next(
        Variants.from_gt_array(
            gt_array, samples=create_sample_names(gt_array)
        ).iter_vars_chunks()
    )
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
    variants = Variants.from_gt_array(gts, samples=["a", "b", "c", "d"])

    expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.0]
    dists = calc_pairwise_kosman_dists(variants)
    assert numpy.allclose(dists.dist_vector, expected)

    dists_emb = calc_pairwise_kosman_dists(
        variants, use_approx_embedding_algorithm=True
    )
    dists = dists.square_dists
    dists_emb = dists_emb.square_dists
    dists_emb = dists_emb.loc[dists.index, :].loc[:, dists.index]
    assert numpy.corrcoef(dists_emb.values.flat, dists.values.flat)[0, 1] > 0.99

    dists_emb = calc_pairwise_kosman_dists(
        variants, use_approx_embedding_algorithm=True, num_threads=2
    )
    dists_emb = dists_emb.square_dists
    dists_emb = dists_emb.loc[dists.index, :].loc[:, dists.index]
    assert numpy.corrcoef(dists_emb.values.flat, dists.values.flat)[0, 1] > 0.99


def test_euclidean_dists():
    num_samples = 4
    num_traits = 10
    numpy.random.seed(42)
    samples = pandas.DataFrame(numpy.random.uniform(size=(num_samples, num_traits)))
    dists = calc_pairwise_euclidean_dists(samples)
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


def test_kosman_pairwise_with_filtered_vars():
    # the chunks are asked for once per distance calculation, and the
    # embedding algorithm asks for them several times, so this used to give
    # wrong distances, or to fail, when the variants came from a filter
    rng = numpy.random.default_rng(7)
    num_samples = 30
    gts = rng.integers(0, 2, size=(60, num_samples, 2))
    samples = [f"sample_{idx}" for idx in range(num_samples)]

    def create_vars():
        variants = Variants.from_gt_array(gts, samples=samples)
        variants.desired_num_vars_per_chunk = 10
        return variants

    expected = calc_pairwise_kosman_dists(create_vars()).dist_vector

    filtered_vars = filter_by_missing_data(create_vars(), max_allowed_missing_rate=1)
    # filtering nothing out has to leave the distances untouched, no matter how
    # many times the filtered variants are used
    assert numpy.allclose(
        calc_pairwise_kosman_dists(filtered_vars).dist_vector, expected
    )
    assert numpy.allclose(
        calc_pairwise_kosman_dists(filtered_vars).dist_vector, expected
    )

    random.seed(0)
    dists_emb = calc_pairwise_kosman_dists(
        filtered_vars, use_approx_embedding_algorithm=True
    )
    square_dists_emb = dists_emb.square_dists
    assert square_dists_emb.shape == (num_samples, num_samples)
    assert not numpy.any(numpy.isnan(square_dists_emb.values))


def test_the_pop_sample_order_is_the_one_that_was_asked_for():
    """The rows used to come out in the order the samples have in the chunk
    while being labelled in the order they were asked for, so asking for them
    in another order put the distances of one sample under the name of
    another."""
    rng = numpy.random.default_rng(1)
    gts = rng.integers(0, 2, (200, 4, 2)).astype(numpy.int8)
    chunk = next(
        Variants.from_gt_array(gts, samples=["a", "b", "c", "d"]).iter_vars_chunks()
    )

    straight, snps1 = _calc_kosman_dist_for_chunk(
        chunk, pop1_samples=["a", "b"], pop2_samples=["c", "d"]
    )
    swapped, snps2 = _calc_kosman_dist_for_chunk(
        chunk, pop1_samples=["b", "a"], pop2_samples=["c", "d"]
    )
    assert list(straight.index) == ["a", "b"]
    assert list(swapped.index) == ["b", "a"]
    # whatever order they are asked in, a row belongs to the sample it names
    for sample in ("a", "b"):
        assert numpy.allclose(straight.loc[sample].values, swapped.loc[sample].values)
        assert numpy.allclose(snps1.loc[sample].values, snps2.loc[sample].values)

    # and the columns too
    cols_swapped, _ = _calc_kosman_dist_for_chunk(
        chunk, pop1_samples=["a", "b"], pop2_samples=["d", "c"]
    )
    for sample in ("c", "d"):
        assert numpy.allclose(straight[sample].values, cols_swapped[sample].values)


def test_the_kosman_dists_are_the_ones_the_pairwise_loop_gave():
    """The distances are worked out for every pair at once with matrix
    products now. This pins them to what comparing the pairs one by one gave,
    over multiallelic variants and missing genotypes, half missing included."""
    rng = numpy.random.default_rng(4)
    num_vars, num_samples = 120, 9
    gts = rng.integers(0, 4, (num_vars, num_samples, 2)).astype(numpy.int8)
    gts[rng.random(gts.shape) < 0.2] = -1
    samples = [f"s{idx}" for idx in range(num_samples)]
    chunk = next(Variants.from_gt_array(gts, samples=samples).iter_vars_chunks())

    dist_sums, n_snps = _calc_kosman_dist_for_chunk(chunk)

    # the same thing, one pair at a time, straight from the definition
    expected_dists, expected_snps = [], []
    for i, j in itertools.combinations(range(num_samples), 2):
        total, counted = 0.0, 0
        for var in range(num_vars):
            gt_i, gt_j = gts[var, i], gts[var, j]
            if -1 in gt_i or -1 in gt_j:
                continue
            counted += 1
            shared = set(gt_i.tolist()) & set(gt_j.tolist())
            if set(gt_i.tolist()) == set(gt_j.tolist()):
                total += 0
            elif not shared:
                total += 1
            else:
                total += 0.5
        expected_dists.append(total)
        expected_snps.append(counted)

    assert numpy.allclose(dist_sums, expected_dists)
    assert numpy.array_equal(n_snps, expected_snps)
