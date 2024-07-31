import math
from typing import Sequence
import itertools

import numpy
import pandas

from pynei.pipeline import Pipeline
from pynei.config import MISSING_ALLELE


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


class _KosmanDistCalculator:
    def __init__(self, chunk):
        """It calculates the pairwise distance between individuals using the Kosman-Leonard dist

        The Kosman distance is explained in "Similarity coefficients for molecular markers in
        studies of genetic relationships between individuals for haploid, diploid, and polyploid
        species"
        Kosman, Leonard (2005) Mol. Ecol. (DOI: 10.1111/j.1365-294X.2005.02416.x)
        """

        self.indi_names = chunk.samples
        gt_array = chunk.gts.gt_array
        self.gt_array = gt_array
        self.allele_is_missing = gt_array == MISSING_ALLELE

    def _get_sample_gts(self, indi_i, indi_j):
        gt_i = self.gt_array[:, indi_i, :]
        is_missing_i = numpy.sum(self.allele_is_missing[:, indi_i, :], axis=1) > 0

        gt_j = self.gt_array[:, indi_j, :]
        is_missing_j = numpy.sum(self.allele_is_missing[:, indi_j, :], axis=1) > 0

        is_called = numpy.logical_not(numpy.logical_or(is_missing_i, is_missing_j))

        gt_i = gt_i[is_called, ...]
        gt_j = gt_j[is_called, ...]
        return gt_i, gt_j

    def calc_dist_btw_two_indis(self, indi_i, indi_j):
        dist_sum, n_snps = self.calc_dist_sum_and_n_snps_btw_two_indis(indi_i, indi_j)
        return dist_sum / n_snps

    def calc_dist_sum_and_n_snps_btw_two_indis(self, indi_i, indi_j):
        gt_i, gt_j = self._get_sample_gts(indi_i, indi_j)

        if gt_i.shape[1] != 2:
            raise ValueError("Only diploid are allowed")

        alleles_comparison1 = gt_i == gt_j.transpose()[:, :, None]
        alleles_comparison2 = gt_j == gt_i.transpose()[:, :, None]

        result = numpy.add(
            numpy.any(alleles_comparison2, axis=2).sum(axis=0),
            numpy.any(alleles_comparison1, axis=2).sum(axis=0),
        )

        result2 = numpy.full(result.shape, fill_value=0.5)
        result2[result == 0] = 1
        result2[result == 4] = 0
        return result2.sum(), result2.shape[0]

    @property
    def num_items(self):
        return len(self.indi_names)


def _calc_pairwise_dists_between_pops(
    dist_between_items_calculator,
    pop1_samples=None,
    pop2_samples=None,
):
    if (pop1_samples is not None and pop2_samples is None) or (
        pop1_samples is None and pop2_samples is not None
    ):
        raise ValueError(
            "When pop1_samples or pop2_samples are given both should be given"
        )

    if pop1_samples is None:
        n_samples = dist_between_items_calculator.num_items
        num_dists_to_calculate = int((n_samples**2 - n_samples) / 2)
        dists_sum = numpy.zeros(num_dists_to_calculate)
        n_snps_matrix = numpy.zeros(num_dists_to_calculate)
    else:
        shape = (len(pop1_samples), len(pop2_samples))
        dists_sum = numpy.zeros(shape)
        n_snps_matrix = numpy.zeros(shape)

    indi_names = dist_between_items_calculator.indi_names
    calc_dist_between_two_indis = (
        dist_between_items_calculator.calc_dist_sum_and_n_snps_btw_two_indis
    )

    if pop1_samples is None:
        sample_combinations = itertools.combinations(range(n_samples), 2)
    else:
        pop1_indi_idxs = [
            idx for idx, sample in enumerate(indi_names) if sample in pop1_samples
        ]
        pop2_indi_idxs = [
            idx for idx, sample in enumerate(indi_names) if sample in pop2_samples
        ]
        sample_combinations = itertools.product(pop1_indi_idxs, pop2_indi_idxs)

    index = 0
    for sample_i, sample_j in sample_combinations:
        dist_sum, n_snps = calc_dist_between_two_indis(sample_i, sample_j)

        if pop1_samples is None:
            dists_sum[index] = dist_sum
            n_snps_matrix[index] = n_snps
            index += 1
        else:
            dists_samplei_idx = pop1_indi_idxs.index(sample_i)
            dists_samplej_idx = pop2_indi_idxs.index(sample_j)
            dists_sum[dists_samplei_idx, dists_samplej_idx] = dist_sum
            n_snps_matrix[dists_samplei_idx, dists_samplej_idx] = n_snps

    return dists_sum, n_snps_matrix


def _calc_pairwise_dists(
    dist_between_items_calculator,
    use_approx_embedding_algorithm=False,
):
    if use_approx_embedding_algorithm:
        dists_between_all_indis_and_some_ref_indis = (
            _calc_pairwise_dists_using_embedding(dist_between_items_calculator)
        )
        dist_calculator = _EuclideanCalculator(
            dists_between_all_indis_and_some_ref_indis,
        )
        dists = _calc_pairwise_dists_between_pops(dist_calculator)
    else:
        dists = _calc_pairwise_dists_between_pops(dist_between_items_calculator)
    return dists


def _calc_kosman_dist_for_chunk(chunk, use_approx_embedding_algorithm=False):
    dist_between_items_calculator = _KosmanDistCalculator(chunk)
    return _calc_pairwise_dists(
        dist_between_items_calculator,
        use_approx_embedding_algorithm=use_approx_embedding_algorithm,
    )


def _reduce_kosman_dists(acummulated_dists_and_snps, new_dists_and_snps):
    new_dists, new_n_snps = new_dists_and_snps
    if acummulated_dists_and_snps is None:
        abs_distances = new_dists_and_snps[0].copy()
        n_snps_matrix = new_dists_and_snps[1]
    else:
        abs_distances, n_snps_matrix = acummulated_dists_and_snps
        abs_distances = numpy.add(abs_distances, new_dists)
        n_snps_matrix = numpy.add(n_snps_matrix, new_n_snps)
    return abs_distances, n_snps_matrix


def calc_pairwise_kosman_dists(
    variants, min_num_snps=None, num_processes=2
) -> Distances:
    """It calculates the distance between individuals using the Kosman
    distance.

    The Kosman distance is explained in DOI: 10.1111/j.1365-294X.2005.02416.x
    """
    samples = variants.samples

    pipeline = Pipeline(
        map_functs=[_calc_kosman_dist_for_chunk], reduce_funct=_reduce_kosman_dists
    )
    res = pipeline.map_and_reduce(variants, num_processes=num_processes)
    abs_distances, n_snps_matrix = res

    if min_num_snps is not None:
        n_snps_matrix[n_snps_matrix < min_num_snps] = numpy.nan

    with numpy.errstate(invalid="ignore"):
        dists = abs_distances / n_snps_matrix
    dists = Distances(dists, samples)
    return dists
