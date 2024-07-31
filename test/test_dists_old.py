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
