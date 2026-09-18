from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


from pynei.config import MIN_NUM_SAMPLES_FOR_POP_STAT, DEF_POLY_THRESHOLD
from pynei.utils_pop import Pops, _calc_pops_idxs
from pynei.utils_stats import (
    StatsDistrib,
    _prepare_bins,
    _summarize_pop_dframe,
    _add_distrib_contributions,
    _finish_distrib,
)
from pynei.gt_counts import _calc_obs_het_per_var, _calc_maf_per_var
from pynei.diversity import (
    PolyVarsStats,
    _calc_exp_het_per_var,
    _calc_unbiased_exp_het_per_var,
    _calc_num_poly_vars,
)
from pynei.pipeline import run_chunk_calcs


class PerVarStat(StrEnum):
    """The statistics that calc_per_var_distribs can calculate.

    The value of every member is the name of the field of PerVarDistribs that
    holds its result.
    """

    OBS_HET = "obs_het"
    "Observed heterozygosity of every variant"

    MAF = "maf"
    "Major allele frequency of every variant"

    EXP_HET = "exp_het"
    "Expected heterozygosity of every variant"

    POLY_VARS_RATIO = "poly_vars_ratio"
    "How many of the variants are polymorphic"


@dataclass(frozen=True)
class PerVarDistribs:
    """The result of calc_per_var_distribs.

    A field is None when its statistic was not asked for.
    """

    obs_het: StatsDistrib | None = None
    "The distribution of the observed heterozygosity per variant, per pop"

    maf: StatsDistrib | None = None
    "The distribution of the major allele frequency per variant, per pop"

    exp_het: StatsDistrib | None = None
    "The distribution of the expected heterozygosity per variant, per pop"

    poly_vars_ratio: PolyVarsStats | None = None
    "How many of the variants are polymorphic, per pop"


class _DistribCalc:
    # One statistic calculated for every variant of the chunk, summarized
    # into the contribution of the chunk to its StatsDistrib

    def __init__(self, calc_stat_per_var, hist_bin_edges):
        self.calc_stat_per_var = calc_stat_per_var
        self.hist_bin_edges = hist_bin_edges

    def calc_for_chunk(self, chunk, cache):
        return _summarize_pop_dframe(
            self.calc_stat_per_var(chunk, cache), self.hist_bin_edges
        )

    def reduce(self, accumulated, contribution):
        return _add_distrib_contributions(accumulated, contribution)

    def finish(self, accumulated):
        return _finish_distrib(accumulated, self.hist_bin_edges)


class _PolyVarsRatioCalc:
    def __init__(self, pops, poly_threshold, min_num_samples):
        self.pops = pops
        self.poly_threshold = poly_threshold
        self.min_num_samples = min_num_samples

    def calc_for_chunk(self, chunk, cache):
        return _calc_num_poly_vars(
            chunk,
            poly_threshold=self.poly_threshold,
            pops=self.pops,
            min_num_samples=self.min_num_samples,
            cache=cache,
        )

    def reduce(self, accumulated, contribution):
        return {
            name: accumulated[name] + values for name, values in contribution.items()
        }

    def finish(self, accumulated):
        num_poly = accumulated["num_poly"]
        num_variable = accumulated["num_variable"]
        num_with_data = accumulated["tot_num_variants_with_data"]
        return PolyVarsStats(
            num_poly=num_poly,
            poly_ratio=num_poly / num_with_data,
            poly_ratio_over_variables=num_poly / num_variable,
            num_variable=num_variable,
            tot_num_variants_with_data=num_with_data,
        )


def _normalize_stats(stats) -> frozenset[PerVarStat]:
    if isinstance(stats, str):
        # a str is iterable, but it is meant as one stat, not as its letters
        stats = [stats]
    # PerVarStat is a StrEnum, so this takes the strings and the members, and
    # it raises for anything else
    stats = frozenset(PerVarStat(stat) for stat in stats)
    if not stats:
        raise ValueError("At least one stat should be asked for")
    return stats


def calc_per_var_distribs(
    variants,
    stats: Iterable[PerVarStat | str] = tuple(PerVarStat),
    pops: Pops | None = None,
    min_num_samples: int = MIN_NUM_SAMPLES_FOR_POP_STAT,
    hist_kwargs: dict | None = None,
    unbiased_exp_het: bool = True,
    ploidy: int | None = None,
    poly_threshold: float = DEF_POLY_THRESHOLD,
    num_processes: int = 1,
) -> PerVarDistribs:
    """It calculates several per variant statistics in one pass over the variants.

    Every statistic is calculated for every variant, and what is given back is
    its distribution: the mean and a histogram, per pop. By default all the
    statistics in PerVarStat are calculated, and asking for fewer of them is
    only an optimization, they share the allele counts and the het masks, so
    doing them together costs little more than doing one.

    unbiased_exp_het and ploidy only matter for exp_het, and poly_threshold
    only matters for poly_vars_ratio.
    """
    stats = _normalize_stats(stats)
    pops_idxs = _calc_pops_idxs(pops, variants.samples)
    hist_bin_edges = _prepare_bins(hist_kwargs, default_range=(0, 1))

    calcs = {}
    if PerVarStat.OBS_HET in stats:

        def calc_obs_het(chunk, cache):
            return _calc_obs_het_per_var(chunk, pops=pops_idxs, cache=cache)[
                "obs_het_per_var"
            ]

        calcs[PerVarStat.OBS_HET] = _DistribCalc(calc_obs_het, hist_bin_edges)

    if PerVarStat.MAF in stats:

        def calc_maf(chunk, cache):
            return _calc_maf_per_var(
                chunk, pops=pops_idxs, min_num_samples=min_num_samples, cache=cache
            )["major_allele_freqs_per_var"]

        calcs[PerVarStat.MAF] = _DistribCalc(calc_maf, hist_bin_edges)

    if PerVarStat.EXP_HET in stats:
        calc_exp_het_per_var = (
            _calc_unbiased_exp_het_per_var
            if unbiased_exp_het
            else _calc_exp_het_per_var
        )

        def calc_exp_het(chunk, cache):
            return calc_exp_het_per_var(
                chunk,
                pops=pops_idxs,
                min_num_samples=min_num_samples,
                ploidy=ploidy,
                cache=cache,
            )["exp_het"]

        calcs[PerVarStat.EXP_HET] = _DistribCalc(calc_exp_het, hist_bin_edges)

    if PerVarStat.POLY_VARS_RATIO in stats:
        calcs[PerVarStat.POLY_VARS_RATIO] = _PolyVarsRatioCalc(
            pops_idxs, poly_threshold=poly_threshold, min_num_samples=min_num_samples
        )

    results = run_chunk_calcs(variants, calcs, num_processes=num_processes)
    return PerVarDistribs(**{str(stat): result for stat, result in results.items()})
