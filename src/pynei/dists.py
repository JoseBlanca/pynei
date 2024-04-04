import itertools
from typing import Sequence
import math

import numpy
import pandas


def _get_vector_from_square(square_dists):
    num_indis = square_dists.shape[0]
    len_vector = (num_indis**2 - num_indis) // 2
    dist_vector = numpy.empty((len_vector,), dtype=square_dists.dtype)
    current_pos = 0
    for row_idx in range(num_indis):
        this_row_items = square_dists[row_idx, row_idx + 1 :]
        start = current_pos
        stop = current_pos + this_row_items.shape[0]
        dist_vector[start:stop] = this_row_items
        current_pos = stop
    return dist_vector


def _calc_num_indis_from_dist_vector(dist_vector_size):
    a = 1
    b = -1
    c = -2 * dist_vector_size
    num_indis = int((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
    return num_indis


def _get_square_from_vector(dist_vector):
    num_indis = _calc_num_indis_from_dist_vector(dist_vector.size)
    square = numpy.zeros((num_indis, num_indis), dtype=dist_vector.dtype)

    col_start = 1
    vector_start = 0
    for row_idx in range(num_indis):
        num_row_items = num_indis - row_idx - 2
        col_stop = col_start + num_row_items + 1
        vector_stop = vector_start + num_row_items + 1
        items = dist_vector[vector_start:vector_stop]
        square[row_idx, col_start:col_stop] = items

        reversed_items = items
        r_col_idx = row_idx
        r_row_start = col_start
        r_row_stop = col_stop
        square[r_row_start:r_row_stop, r_col_idx] = reversed_items

        col_start += 1
        vector_start = vector_stop
    return square


class Distances:
    def __init__(
        self,
        dist_vector: numpy.ndarray,
        names: Sequence[str] | Sequence[int] | None = None,
    ):
        self.dist_vector = numpy.array(dist_vector)
        self.dist_vector.flags.writeable = False

        expected_num_indis = _calc_num_indis_from_dist_vector(self.dist_vector.shape[0])
        if names is None:
            names = numpy.arange(expected_num_indis)
        else:
            names = numpy.array(names)
            if names.size != expected_num_indis:
                raise ValueError(
                    f"Expected num indis ({expected_num_indis}) does not match the given number of names ({names.shape})"
                )
        names.flags.writeable = False
        self.names = names

    @classmethod
    def from_square_dists(cls, dists: pandas.DataFrame):
        if dists.shape[0] != dists.shape[1]:
            raise ValueError(
                f"A square dist matrix is required, but shape was not squared: {dists.shape}"
            )
        names = numpy.array(dists.index)
        dist_vector = _get_vector_from_square(dists.values)
        return cls(dist_vector=dist_vector, names=names)

    @property
    def square_dists(self):
        dists = _get_square_from_vector(self.dist_vector)
        dists = pandas.DataFrame(dists, index=self.names, columns=self.names)
        return dists

    @property
    def triang_list_of_lists(self):
        dist_vector = iter(self.dist_vector)
        length = 0
        dists = []
        while True:
            dist_row = list(itertools.islice(dist_vector, length))
            if length and not dist_row:
                break
            dist_row.append(0)
            dists.append(dist_row)
            length += 1
        return dists
