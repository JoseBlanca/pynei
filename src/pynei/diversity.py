from dataclasses import dataclass

import numpy
import pandas

from pynei.config import (
    MIN_NUM_SAMPLES_FOR_POP_STAT,
    DEF_POLY_THRESHOLD,
)
from pynei.gt_counts import _count_alleles_per_var, _calc_maf_per_var
from pynei.utils_pop import Pops


def _calc_exp_het_per_var(
    chunk, pops, min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT, ploidy=None, cache=None
):
    if ploidy is None:
        ploidy = chunk.ploidy

    res = _count_alleles_per_var(
        chunk,
        pops=pops,
        calc_freqs=True,
        min_num_samples=min_num_samples,
        cache=cache,
    )

    sorted_pops = sorted(pops.keys())

    missing_allelic_gts = {
        pop_id: res["counts"][pop_id]["missing_gts_per_var"] for pop_id in sorted_pops
    }
    missing_allelic_gts = pandas.DataFrame(missing_allelic_gts, columns=sorted_pops)

    exp_het = {}
    for pop_id in sorted_pops:
        allele_freqs = res["counts"][pop_id]["allelic_freqs"].values
        exp_het[pop_id] = 1 - numpy.sum(allele_freqs**ploidy, axis=1)
    exp_het = pandas.DataFrame(exp_het, columns=sorted_pops)

    return {"exp_het": exp_het, "missing_allelic_gts": missing_allelic_gts}


def _calc_unbiased_exp_het_per_var(
    chunk, pops, min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT, ploidy=None, cache=None
):
    "Calculated using Unbiased Heterozygosity (Codom Data) Genalex formula"
    if ploidy is None:
        ploidy = chunk.ploidy

    res = _calc_exp_het_per_var(
        chunk,
        pops=pops,
        min_num_samples=min_num_samples,
        ploidy=ploidy,
        cache=cache,
    )
    exp_het = res["exp_het"]

    missing_allelic_gts = res["missing_allelic_gts"]

    num_allelic_gtss = []
    for pop in missing_allelic_gts.columns:
        pop_slice = pops[pop]
        if (
            isinstance(pop_slice, slice)
            and pop_slice.start is None
            and pop_slice.stop is None
            and pop_slice.step is None
        ):
            num_allelic_gts = chunk.num_samples * ploidy
        else:
            num_allelic_gts = len(pops[pop]) * ploidy
        num_allelic_gtss.append(num_allelic_gts)
    num_exp_allelic_gts_per_pop = numpy.array(num_allelic_gtss)
    num_called_allelic_gts_per_snp = (
        num_exp_allelic_gts_per_pop[numpy.newaxis, :] - missing_allelic_gts
    )
    num_samples = num_called_allelic_gts_per_snp / ploidy

    unbiased_exp_het = (2 * num_samples / (2 * num_samples - 1)) * exp_het
    return {
        "exp_het": unbiased_exp_het,
        "missing_allelic_gts": missing_allelic_gts,
    }


@dataclass(frozen=True)
class PolyVarsStats:
    """How many of the variants are polymorphic, per pop."""

    num_poly: pandas.Series
    "How many variants have a major allele freq below the poly threshold"

    poly_ratio: pandas.Series
    "num_poly over the variants that have enough data"

    poly_ratio_over_variables: pandas.Series
    "num_poly over the variants that are not fixed"

    num_variable: pandas.Series
    "How many variants have a major allele freq below 1"

    tot_num_variants_with_data: pandas.Series
    "How many variants had enough data to be counted"


def _calc_num_poly_vars(
    chunk,
    poly_threshold=DEF_POLY_THRESHOLD,
    pops: Pops | None = None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
    cache=None,
):
    res = _calc_maf_per_var(
        chunk,
        pops=pops,
        min_num_samples=min_num_samples,
        cache=cache,
    )
    mafs = res["major_allele_freqs_per_var"]

    num_not_nas = mafs.notna().sum(axis=0)
    num_variable = (mafs < 1).sum(axis=0)
    num_poly = (mafs < poly_threshold).sum(axis=0)
    res = {
        "num_poly": num_poly,
        "num_variable": num_variable,
        "tot_num_variants_with_data": num_not_nas,
    }
    return res
