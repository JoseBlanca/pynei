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


def _count_each_allele(gts, alleles):
    """How often every allele appears in every variant, and how many are missing.

    It is one pass over the genotypes per allele, comparing and counting, and
    one more for the missing ones. That is the fastest way there is for the
    two or three alleles a variant has: bincount over the row and the allele
    was 3x slower, because its index has to be 8 bytes per genotype, add.at
    was 27x slower and a one hot sum 9x. What used to be slow was not the
    counting but finding out which alleles to count, see the caller.
    """
    counts = numpy.empty((gts.shape[0], len(alleles)), dtype=numpy.int32)
    for idx, allele in enumerate(alleles):
        counts[:, idx] = numpy.count_nonzero(gts == allele, axis=(1, 2))
    missing = numpy.count_nonzero(gts == MISSING_ALLELE, axis=(1, 2))
    return counts, missing


def _count_alleles_per_var(
    chunk,
    calc_freqs: bool,
    pops: dict[str, list[int]] | None = None,
    alleles=None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
    cache=None,
):
    """The allele counts, per variant and per pop.

    The columns are the alleles the chunk has, or the given ones. Giving them
    is what makes the counts of different chunks line up, and a chunk with an
    allele that was not given is an error rather than a count left out.
    """
    if cache is not None:
        alleles_key = None if alleles is None else tuple(alleles)
        key = (
            "allele_counts",
            calc_freqs,
            _pops_key(pops),
            alleles_key,
            min_num_samples,
        )
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
    ploidy = chunk.ploidy

    if pops is None:
        pops = {DEF_POP_NAME: slice(None, None)}

    if alleles is None:
        # every allele up to the biggest one is counted, and the ones in
        # between that turn out not to be there are dropped at the end. The
        # biggest one is a 0.1 ms reduction, while asking numpy.unique which
        # alleles there are cost more than all the counting put together
        max_allele = int(gts.max()) if gts.size else MISSING_ALLELE
        alleles_to_count = list(range(max_allele + 1))
    else:
        alleles_to_count = list(alleles)

    counts_per_pop = {}
    present = numpy.zeros(len(alleles_to_count), dtype=bool)
    for pop_id, pop_slice in pops.items():
        pop_gts = gts[:, pop_slice, :]
        counts, missing = _count_each_allele(pop_gts, alleles_to_count)
        present |= counts.any(axis=0)

        # every genotype is either one of the alleles counted or missing, so
        # if the counts do not add up some value is neither
        num_gts_per_var = pop_gts.shape[1] * pop_gts.shape[2]
        if numpy.any(counts.sum(axis=1) + missing != num_gts_per_var):
            found = set(numpy.unique(pop_gts).tolist()).difference(
                alleles_to_count, [MISSING_ALLELE]
            )
            if alleles is None:
                raise ValueError(
                    f"There are genotypes below the missing allele "
                    f"({MISSING_ALLELE}): {sorted(found)}"
                )
            raise RuntimeError(
                f"These gts have alleles ({sorted(found)}) not present in the "
                f"given ones ({alleles_to_count})"
            )
        counts_per_pop[pop_id] = (counts, missing, num_gts_per_var)

    alleles_in_chunk = {alleles_to_count[idx] for idx in numpy.flatnonzero(present)}
    if alleles is None:
        kept = numpy.flatnonzero(present)
        columns = [alleles_to_count[idx] for idx in kept]
    else:
        kept = slice(None)
        columns = alleles_to_count

    result = {}
    for pop_id, (counts, missing_counts, num_gts_per_var) in counts_per_pop.items():
        allele_counts = pandas.DataFrame(counts[:, kept], columns=columns)
        result[pop_id] = {
            "allele_counts": allele_counts,
            "missing_gts_per_var": missing_counts,
        }

        if calc_freqs:
            num_allelic_gts_per_snp = num_gts_per_var - missing_counts
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
