from typing import Sequence

import numpy
import pandas

from .genotypes import Genotypes, _get_pop_masks
from .config import (
    MIN_NUM_GENOTYPES_FOR_POP_STAT,
    MISSING_ALLELE,
    DEF_POLY_THRESHOLD,
)


def calc_major_allele_freqs(
    gts: Genotypes,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
) -> pandas.DataFrame:
    res = gts._count_alleles_per_var(
        pops=pops,
        calc_freqs=True,
        min_num_genotypes=min_num_genotypes,
    )
    freqs = []
    pops = sorted(res.keys())
    for pop in pops:
        freqs_for_pop = res[pop]["allelic_freqs"].max(axis=1)
        freqs.append(freqs_for_pop.values)
    freqs = pandas.DataFrame(numpy.array(freqs).T, columns=pops)
    return freqs


def calc_poly_vars_ratio(
    gts: Genotypes,
    poly_threshold=DEF_POLY_THRESHOLD,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
):
    mafs = calc_major_allele_freqs(
        gts=gts, pops=pops, min_num_genotypes=min_num_genotypes
    )
    num_not_nas = mafs.notna().sum(axis=0)
    num_poly = (mafs < poly_threshold).sum(axis=0)
    poly_ratio = num_poly / num_not_nas
    return poly_ratio


def _calc_obs_het_per_var(
    gts,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
):
    pop_masks = _get_pop_masks(gts, pops)
    gt_array = gts._gt_array
    freqs = []
    pops = []
    num_gts = []
    for pop_id, pop_mask in pop_masks:
        pops.append(pop_id)
        gts_for_pop = gt_array[:, pop_mask, :]
        gt_is_missing = numpy.any(gts_for_pop == MISSING_ALLELE, axis=2)

        first_haploid_gt = gts_for_pop[:, :, 0]
        is_het = None
        for idx in range(1, gts.ploidy):
            haploid_gt = gts_for_pop[:, :, idx]
            different_allele = first_haploid_gt != haploid_gt
            if is_het is None:
                is_het = different_allele
            else:
                is_het = numpy.logical_or(is_het, different_allele)
        het_and_not_missing = numpy.logical_and(is_het, ~gt_is_missing)
        num_obs_het = het_and_not_missing.sum(axis=1)
        num_indis = het_and_not_missing.shape[1]
        num_gts_per_var = num_indis - gt_is_missing.sum(axis=1)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            freq_obs_het = num_obs_het / num_gts_per_var
        not_enough_indis = num_gts_per_var < min_num_genotypes
        freq_obs_het[not_enough_indis] = numpy.nan
        freqs.append(freq_obs_het)
        num_gts.append(num_gts_per_var)
    freqs = pandas.DataFrame(numpy.array(freqs).T, columns=pops)
    num_gts = pandas.DataFrame(numpy.array(num_gts).T, columns=pops)
    return {"freqs": freqs, "num_gts": num_gts}


def calc_obs_het(
    gts: Genotypes,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
):
    res = _calc_obs_het_per_var(
        gts,
        pops=pops,
        min_num_genotypes=min_num_genotypes,
    )
    freqs = res["freqs"]
    return freqs.mean(axis=0)


def _calc_exp_het_per_var(
    gts: Genotypes,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    unbiased=True,
):
    res = gts._count_alleles_per_var(
        pops=pops, calc_freqs=True, min_num_genotypes=min_num_genotypes
    )
    pop_names = sorted(res.keys())
    nei = []
    for pop in pop_names:
        pop_freqs = res[pop]["allelic_freqs"]
        nei_per_var = 1 - numpy.sum((pop_freqs**2).values, axis=1)
        if unbiased:
            num_indis = len(pops[pop])
            num_indis_per_var = num_indis - (
                res[pop]["missing_gts_per_var"] / gts.ploidy
            )
            nei_per_var = (
                (2 * num_indis_per_var) / (2 * num_indis_per_var - 1)
            ) * nei_per_var
        nei.append(nei_per_var)

    nei = pandas.DataFrame(numpy.array(nei).T, columns=pop_names)
    return {"exp_het": nei, "counts_and_freqs_per_var": res}


def calc_exp_het(
    gts: Genotypes,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    unbiased=True,
):
    nei_per_var = _calc_exp_het_per_var(
        gts=gts, pops=pops, min_num_genotypes=min_num_genotypes, unbiased=unbiased
    )["exp_het"]
    nei = pandas.Series(
        numpy.nanmean(nei_per_var.values, axis=0), index=nei_per_var.columns
    )
    return nei


calc_nei_diversity = calc_exp_het
