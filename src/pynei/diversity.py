from functools import partial

import numpy
import pandas

from pynei.config import MIN_NUM_SAMPLES_FOR_POP_STAT
from pynei.gt_counts import _count_alleles_per_var
from pynei.utils_pop import _calc_pops_idxs
from pynei.utils_stats import _calc_stats_per_var


def _calc_exp_het_per_var(
    chunk, pops, min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT, ploidy=None
):
    if ploidy is None:
        ploidy = chunk.ploidy

    res = _count_alleles_per_var(
        chunk,
        pops=pops,
        calc_freqs=True,
        min_num_samples=min_num_samples,
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
    chunk, pops, min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT, ploidy=None
):
    "Calculated using Unbiased Heterozygosity (Codom Data) Genalex formula"
    if ploidy is None:
        ploidy = chunk.ploidy

    res = _calc_exp_het_per_var(
        chunk,
        pops=pops,
        min_num_samples=min_num_samples,
        ploidy=ploidy,
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


def calc_exp_het_stats_per_var(
    variants,
    pops: dict[str, list[str]] | None = None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
    ploidy=None,
    hist_kwargs=None,
    unbiased=True,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["range"] = hist_kwargs.get("range", (0, 1))

    samples = variants.samples
    pops = _calc_pops_idxs(pops, samples)

    if unbiased:
        calc_het_funct = _calc_unbiased_exp_het_per_var
    else:
        calc_het_funct = _calc_exp_het_per_var

    return _calc_stats_per_var(
        variants=variants,
        calc_stats_for_chunk=partial(
            calc_het_funct,
            pops=pops,
            min_num_samples=min_num_samples,
            ploidy=ploidy,
        ),
        get_stats_for_chunk_result=lambda x: x["exp_het"],
        hist_kwargs=hist_kwargs,
    )
