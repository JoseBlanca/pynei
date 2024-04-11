import itertools
from typing import Sequence
import math
import warnings


import numpy
import pandas

from .config import MIN_NUM_GENOTYPES_FOR_POP_STAT, MISSING_ALLELE
from .stats import _calc_exp_het_per_var, _calc_obs_het_per_var
from .genotypes import Genotypes


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


def hmean(array, axis=0, dtype=None):
    # Harmonic mean only defined if greater than zero
    if isinstance(array, numpy.ma.MaskedArray):
        size = array.count(axis)
    else:
        if axis is None:
            array = array.ravel()
            size = array.shape[0]
        else:
            size = array.shape[axis]
    with numpy.errstate(divide="ignore"):
        inverse_mean = numpy.sum(1.0 / array, axis=axis, dtype=dtype)
    is_inf = numpy.logical_not(numpy.isfinite(inverse_mean))
    hmean = size / inverse_mean
    hmean[is_inf] = numpy.nan

    return hmean


def _calc_pairwise_dest(
    gts,
    pops,
    pop1,
    pop2,
    exp_het,
    obs_het,
    counts_and_freqs_per_var,
    min_num_genotypes,
):
    debug = False

    num_pops = 2
    ploidy = gts.ploidy

    exp_het1 = exp_het.loc[:, pop1].values
    exp_het2 = exp_het.loc[:, pop2].values
    hs_per_var = (exp_het1 + exp_het2) / 2.0
    if debug:
        print("exp_het1", exp_het1)
        print("exp_het2", exp_het2)
        print("hs_per_var", hs_per_var)

    allelic_freqs_pop1 = counts_and_freqs_per_var[pop1]["allelic_freqs"].values
    allelic_freqs_pop2 = counts_and_freqs_per_var[pop2]["allelic_freqs"].values
    global_allele_freq = (allelic_freqs_pop1 + allelic_freqs_pop2) / 2.0
    global_exp_het = 1 - numpy.sum(global_allele_freq**ploidy, axis=1)
    ht_per_var = global_exp_het
    if debug:
        print("ht_per_var", ht_per_var)

    obs_het_pop1 = obs_het.loc[:, pop1].values
    obs_het_pop2 = obs_het.loc[:, pop2].values
    if debug:
        print(f"{obs_het_pop1=}")
        print(f"{obs_het_pop2=}")

    num_total_alleles1 = len(pops[pop1]) * ploidy
    called_gts1 = (
        num_total_alleles1
        - counts_and_freqs_per_var[pop1]["missing_gts_per_var"].values
    ) / ploidy
    num_total_alleles2 = len(pops[pop2]) * ploidy
    called_gts2 = (
        num_total_alleles2
        - counts_and_freqs_per_var[pop2]["missing_gts_per_var"].values
    ) / ploidy
    called_gts = numpy.array([called_gts1, called_gts2])
    try:
        called_gts_hmean = hmean(called_gts, axis=0)
    except ValueError:
        called_gts_hmean = None
    if debug:
        print("called_gts_per_pop:", called_gts)

    if called_gts_hmean is None:
        num_vars = gts.shape[0]
        corrected_hs = numpy.full((num_vars,), numpy.nan)
        corrected_ht = numpy.full((num_vars,), numpy.nan)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_obs_het_per_var = numpy.nanmean(
                numpy.array([obs_het_pop1, obs_het_pop2]), axis=0
            )
        corrected_hs = (called_gts_hmean / (called_gts_hmean - 1)) * (
            hs_per_var - (mean_obs_het_per_var / (2 * called_gts_hmean))
        )
        if debug:
            print("mean_obs_het_per_var", mean_obs_het_per_var)
            print("corrected_hs", corrected_hs)
        corrected_ht = (
            ht_per_var
            + (corrected_hs / (called_gts_hmean * num_pops))
            - (mean_obs_het_per_var / (2 * called_gts_hmean * num_pops))
        )
        if debug:
            print("corrected_ht", corrected_ht)

        not_enough_gts = numpy.logical_or(
            called_gts1 < min_num_genotypes, called_gts2 < min_num_genotypes
        )
        corrected_hs[not_enough_gts] = numpy.nan
        corrected_ht[not_enough_gts] = numpy.nan

    num_vars = numpy.count_nonzero(~numpy.isnan(corrected_hs))

    hs = numpy.nansum(corrected_hs)
    ht = numpy.nansum(corrected_ht)
    if debug:
        print(f"{hs=}")
        print(f"{ht=}")

    if num_vars == 0:
        dest = numpy.nan
    else:
        corrected_hs = hs / num_vars
        corrected_ht = ht / num_vars
        dest = (num_pops / (num_pops - 1)) * (
            (corrected_ht - corrected_hs) / (1 - corrected_hs)
        )

    return {"dest": dest}


def calc_jost_dest_dist(
    gts, pops, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT
) -> Distances:
    """Jost's estimate of the Dest differentiation

    This is an implementation of the formulas proposed in GenAlex
    From Genealex documentation:
    Here, Jost’s estimate of differentiation (Dest) (Jost, 2008) is calculated following
    Meirmans and Hedrick eq 2.(2011). Their recommendation to average cHS
    and cHT for estimating Dest across loci is also used. See HS, HT and GST
    below for further details. Note that some software packages estimate Dest
    over loci as the harmonic mean of individual locus Dest values (Meirmans and Hedrick, 2011).

    Jost, L. 2008. GST and its relatives do not measure differentiation. Molecular Ecology 17, 4015-4026.
    Meirmans, PG and Hedrick, PW. 2011. Assessing population structure: FST and related measures. Molecular Ecology Resources 11, 5-18

    """
    pop_ids = sorted(pops.keys())
    num_pops = len(pop_ids)

    dest = pandas.DataFrame(
        numpy.zeros(shape=(num_pops, num_pops), dtype=float),
        columns=pop_ids,
        index=pop_ids,
    )

    res = _calc_exp_het_per_var(
        gts, pops=pops, min_num_genotypes=min_num_genotypes, unbiased=False
    )
    exp_het = res["exp_het"]
    counts_and_freqs_per_var = res["counts_and_freqs_per_var"]
    res_het = _calc_obs_het_per_var(gts, pops=pops, min_num_genotypes=min_num_genotypes)
    obs_het = res_het["freqs"]

    for pop1, pop2 in itertools.combinations(pop_ids, 2):
        res = _calc_pairwise_dest(
            gts,
            pops=pops,
            pop1=pop1,
            pop2=pop2,
            exp_het=exp_het,
            obs_het=obs_het,
            counts_and_freqs_per_var=counts_and_freqs_per_var,
            min_num_genotypes=min_num_genotypes,
        )
        dest.loc[pop1, pop2] = res["dest"]
        dest.loc[pop2, pop1] = res["dest"]
    dest = Distances.from_square_dists(dest)
    return dest


class KosmanDistCalculator:
    def __init__(self, gts: Genotypes):
        """It calculates the pairwise distance between individuals using the Kosman-Leonard dist

        The Kosman distance is explained in "Similarity coefficients for molecular markers in
        studies of genetic relationships between individuals for haploid, diploid, and polyploid
        species"
        Kosman, Leonard (2005) Mol. Ecol. (DOI: 10.1111/j.1365-294X.2005.02416.x)
        """

        self.indi_names = gts.indi_names
        gt_array = gts.gt_array
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

    def _calc_dist_between_two_indis(self, indi_i, indi_j):
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

    def calc_pairwise_dists(self):
        dists = self._calc_pairwise_dists_between_pops()
        return Distances(dists, names=self.indi_names)

    def _calc_pairwise_dists_between_pops(self, pop1_samples=None, pop2_samples=None):
        indi_names = self.indi_names
        if pop1_samples is None:
            n_samples = self.gt_array.shape[1]
            num_dists_to_calculate = int((n_samples**2 - n_samples) / 2)
            dists = numpy.zeros(num_dists_to_calculate)
            n_snps_matrix = numpy.zeros(num_dists_to_calculate)
        else:
            shape = (len(pop1_samples), len(pop2_samples))
            dists = numpy.zeros(shape)
            n_snps_matrix = numpy.zeros(shape)

        dists = numpy.zeros(num_dists_to_calculate)
        n_snps_matrix = numpy.zeros(num_dists_to_calculate)

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
            dist, n_snps = self._calc_dist_between_two_indis(sample_i, sample_j)

            if pop1_samples is None:
                dists[index] = dist
                n_snps_matrix[index] = n_snps
                index += 1
            else:
                dists_samplei_idx = pop1_indi_idxs.index(sample_i)
                dists_samplej_idx = pop2_indi_idxs.index(sample_j)
                dists[dists_samplei_idx, dists_samplej_idx] = dist
                n_snps_matrix[dists_samplei_idx, dists_samplej_idx] = n_snps

        return dists / n_snps_matrix
