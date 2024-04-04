import numpy
import pandas

from pynei.dists import Distances


def test_distances():
    dists_orig = [[0, 0.5, 0.75], [0.5, 0, 0.3], [0.75, 0.3, 0]]
    dists = Distances.from_square_dists(pandas.DataFrame(dists_orig))
    assert numpy.allclose(dists.dist_vector, [0.5, 0.75, 0.3])

    dists = Distances([0.5, 0.75, 0.3])
    assert numpy.allclose(dists.dist_vector, [0.5, 0.75, 0.3])

    assert numpy.allclose(dists.square_dists.values, dists_orig)
