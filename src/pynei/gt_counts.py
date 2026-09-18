import numpy
import pandas

from pynei.config import MIN_NUM_SAMPLES_FOR_POP_STAT, DEF_POP_NAME, MISSING_ALLELE

# The functions in this module take a cache: a dict that lives for one chunk
# and that is shared by every calculation done on it, so that what one of them
# needs and another one has already worked out, like the allele counts or the
# het mask, is done once per chunk.


def _calc_gt_is_missing(chunk, cache=None):
    res = {} if cache is None else cache
    if "gt_is_missing" in res:
        return res

    allele_is_missing = chunk.gts.missing_mask
    res["gt_is_missing"] = numpy.any(allele_is_missing, axis=2)
    return res


def _calc_gt_is_het(chunk, cache=None):
    res = {} if cache is None else cache
    if "gt_is_het" in res:
        return res

    res = _calc_gt_is_missing(chunk, cache=res)
    gt_is_missing = res["gt_is_missing"]

    gt_array = chunk.gts.gt_values
    gt_is_het = numpy.logical_not(
        numpy.all(gt_array == gt_array[:, :, 0][:, :, numpy.newaxis], axis=2)
    )
    res["gt_is_het"] = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))
    return res


def _calc_obs_het_per_var(chunk, pops, cache=None):
    res = _calc_gt_is_het(chunk, cache=cache)
    gt_is_het = res["gt_is_het"]
    gt_is_missing = res["gt_is_missing"]

    obs_het_per_var = {}
    called_gts_per_var = {}
    for pop_name, pop_slice in pops.items():
        num_vars_het_per_var = numpy.sum(gt_is_het[:, pop_slice], axis=1)
        gt_is_missing_for_pop = gt_is_missing[:, pop_slice]
        num_samples = gt_is_missing_for_pop.shape[1]
        num_non_missing_per_var = num_samples - numpy.sum(
            gt_is_missing[:, pop_slice], axis=1
        )
        with numpy.errstate(invalid="ignore"):
            obs_het_per_var[pop_name] = num_vars_het_per_var / num_non_missing_per_var
        called_gts_per_var[pop_name] = num_non_missing_per_var

    obs_het_per_var = pandas.DataFrame(obs_het_per_var)
    called_gts_per_var = pandas.DataFrame(called_gts_per_var)
    return {
        "obs_het_per_var": obs_het_per_var,
        "called_gts_per_var": called_gts_per_var,
    }


def _pops_key(pops):
    # a hashable version of the pops idxs, to use them in a cache key
    if pops is None:
        return None
    return tuple(
        (pop, idxs if isinstance(idxs, slice) else tuple(idxs))
        for pop, idxs in pops.items()
    )


def _count_alleles_per_var(
    chunk,
    calc_freqs: bool,
    pops: dict[str, list[int]] | None = None,
    alleles=None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
    cache=None,
):
    if cache is not None:
        key = ("allele_counts", calc_freqs, _pops_key(pops), min_num_samples)
        if key not in cache:
            cache[key] = _count_alleles_per_var(
                chunk,
                calc_freqs=calc_freqs,
                pops=pops,
                alleles=alleles,
                min_num_samples=min_num_samples,
            )
        return cache[key]

    gts = chunk.gts.gt_values
    missing_mask = chunk.gts.missing_mask

    alleles_in_chunk = set(numpy.unique(gts).tolist()).difference([MISSING_ALLELE])
    alleles = sorted(alleles_in_chunk)
    ploidy = chunk.ploidy

    if pops is None:
        pops = {DEF_POP_NAME: slice(None, None)}

    if alleles is not None:
        if alleles_in_chunk.difference(alleles):
            raise RuntimeError(
                f"These gts have alleles ({alleles_in_chunk}) not present in the given ones ({alleles})"
            )

    result = {}
    for pop_id, pop_slice in pops.items():
        pop_gts = gts[:, pop_slice, :]
        pop_missing_mask = missing_mask[:, pop_slice, :]
        allele_counts = numpy.empty(
            shape=(pop_gts.shape[0], len(alleles)), dtype=numpy.int32
        )
        for idx, allele in enumerate(alleles):
            is_allele = numpy.logical_and(
                pop_gts == allele, numpy.logical_not(pop_missing_mask)
            )
            allele_counts_per_row = numpy.sum(is_allele, axis=(1, 2))
            allele_counts[:, idx] = allele_counts_per_row
        allele_counts = pandas.DataFrame(allele_counts, columns=alleles)
        missing_counts = numpy.sum(pop_missing_mask, axis=(1, 2))

        result[pop_id] = {
            "allele_counts": allele_counts,
            "missing_gts_per_var": missing_counts,
        }

        if calc_freqs:
            expected_num_allelic_gts_in_snp = pop_gts.shape[1] * pop_gts.shape[2]
            num_allelic_gts_per_snp = expected_num_allelic_gts_in_snp - missing_counts
            num_allelic_gts_per_snp = num_allelic_gts_per_snp.reshape(
                (num_allelic_gts_per_snp.shape[0], 1)
            )
            allelic_freqs_per_snp = allele_counts / num_allelic_gts_per_snp
            num_gts_per_snp = (
                num_allelic_gts_per_snp.reshape((num_allelic_gts_per_snp.size,))
                / ploidy
            )
            not_enough_data = num_gts_per_snp < min_num_samples
            allelic_freqs_per_snp[not_enough_data] = numpy.nan

            result[pop_id]["allelic_freqs"] = allelic_freqs_per_snp

    return {"counts": result, "alleles": alleles_in_chunk}


def _calc_maf_per_var(
    chunk,
    pops,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
    cache=None,
):
    res = _count_alleles_per_var(
        chunk,
        pops=pops,
        alleles=None,
        calc_freqs=True,
        min_num_samples=min_num_samples,
        cache=cache,
    )
    major_allele_freqs = {}
    for pop, pop_res in res["counts"].items():
        pop_allelic_freqs = pop_res["allelic_freqs"]
        major_allele_freqs[pop] = pop_allelic_freqs.max(axis=1)
    major_allele_freqs = pandas.DataFrame(major_allele_freqs)
    return {"major_allele_freqs_per_var": major_allele_freqs}
