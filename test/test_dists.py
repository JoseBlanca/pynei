import numpy
import pandas

from pynei.dists import (
    Distances,
    calc_jost_dest_dist,
    calc_kosman_pairwise_dists,
    _KosmanDistCalculator,
    calc_euclidean_pairwise_dists,
)
from pynei import Genotypes


def test_distances():
    dists_orig = [[0, 0.5, 0.75], [0.5, 0, 0.3], [0.75, 0.3, 0]]
    dists = Distances.from_square_dists(pandas.DataFrame(dists_orig))
    assert numpy.allclose(dists.dist_vector, [0.5, 0.75, 0.3])

    dists = Distances([0.5, 0.75, 0.3])
    assert numpy.allclose(dists.dist_vector, [0.5, 0.75, 0.3])

    assert numpy.allclose(dists.square_dists.values, dists_orig)


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
    gts = Genotypes(numpy.array(gts), indi_names=samples)

    pop1 = [1, 2, 3, 4, 5]
    pop2 = [6, 7, 8, 9, 10, 11]
    pops = {"pop1": pop1, "pop2": pop2}

    dists = calc_jost_dest_dist(gts, pops=pops, min_num_genotypes=0)
    assert numpy.allclose(dists.dist_vector, [0.65490196])

    dists = calc_jost_dest_dist(gts, pops=pops, min_num_genotypes=6)
    assert numpy.all(numpy.isnan(dists.dist_vector))


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
    distance = _KosmanDistCalculator(Genotypes(gt_array)).calc_dist_between_two_indis(
        0, 1
    )
    assert distance == 1 / 3

    c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    gt_array = numpy.stack((c, d), axis=1)
    distance = _KosmanDistCalculator(Genotypes(gt_array)).calc_dist_between_two_indis(
        0, 1
    )

    assert distance == 0

    gt_array = numpy.stack((b, d), axis=1)
    distance = _KosmanDistCalculator(Genotypes(gt_array)).calc_dist_between_two_indis(
        0, 1
    )
    assert distance == 0.45


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
    distance_ab = _KosmanDistCalculator(
        Genotypes(gt_array)
    ).calc_dist_between_two_indis(0, 1)

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
    distance_cd = _KosmanDistCalculator(
        Genotypes(gt_array)
    ).calc_dist_between_two_indis(0, 1)

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
    gts = Genotypes(gts, [1, 2, 3, 4])

    expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.0]
    dists = calc_kosman_pairwise_dists(gts)
    assert numpy.allclose(dists.dist_vector, expected)

    dists_emb = calc_kosman_pairwise_dists(gts, use_approx_embedding_algorithm=True)
    dists = dists.square_dists
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


def test_emmbedding():
    num_samples = 40
    num_traits = 10
    numpy.random.seed(42)
    samples = pandas.DataFrame(numpy.random.uniform(size=(num_samples, num_traits)))

    dists = calc_euclidean_pairwise_dists(samples)
    dists_emb = calc_euclidean_pairwise_dists(
        samples, use_approx_embedding_algorithm=True
    )
    dists = dists.square_dists
    dists_emb = dists_emb.square_dists.loc[dists.index, :].loc[:, dists.index]
    assert (
        numpy.corrcoef(dists.values.flat[:10], dists_emb.values.flat[:10])[0, 1] > 0.89
    )
