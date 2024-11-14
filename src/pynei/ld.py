import itertools
from functools import partial
from collections import namedtuple

import numpy

from .config import MISSING_ALLELE

from pynei.config import VAR_TABLE_CHROM_COL, VAR_TABLE_POS_COL

DDOF = 1


def _calc_maf_from_012_gts(gts: numpy.array):
    counts = {}
    for gt in [0, 1, 2]:
        counts[gt] = numpy.sum(gts == gt, axis=1)

    num_gts_per_var = counts[0] + counts[1] + counts[2]
    counts_major_allele = numpy.maximum(counts[0], counts[2])
    freqs_major_allele = counts_major_allele / num_gts_per_var
    freqs_het = counts[1] / num_gts_per_var
    maf_per_var = freqs_major_allele + 0.5 * freqs_het
    return maf_per_var


def _calc_rogers_huff_r2(
    gts1: numpy.ndarray,
    gts2: numpy.ndarray,
    check_no_mafs_above: float | None = 0.95,
    debug=False,
):
    if check_no_mafs_above is not None:
        maf_per_var = _calc_maf_from_012_gts(gts1)
        if numpy.any(maf_per_var > check_no_mafs_above):
            raise ValueError(
                f"There are variations with mafs above {check_no_mafs_above}, filter them out or modify this check"
            )

    covars = numpy.cov(gts1, gts2, ddof=DDOF)
    n_vars1 = gts1.shape[0]
    n_vars2 = gts2.shape[0]
    if debug:
        print("nvars", n_vars1, n_vars2)
    variances = numpy.diag(covars)
    vars1 = variances[:n_vars1]
    vars2 = variances[n_vars1:]
    if debug:
        print("vars1", vars1)
        print("vars2", vars2)

    covars = covars[:n_vars1, n_vars1:]
    if debug:
        print("covars", covars)

    vars1 = numpy.repeat(vars1, n_vars2).reshape((n_vars1, n_vars2))
    vars2 = numpy.tile(vars2, n_vars1).reshape((n_vars1, n_vars2))
    with numpy.errstate(divide="ignore", invalid="ignore"):
        rogers_huff_r = covars / numpy.sqrt(vars1 * vars2)
    # print(vars1)
    # print(vars2)
    return rogers_huff_r


def _chunks_are_close(chunk_pair, max_dist):
    chunk1 = chunk_pair[0]
    chunk2 = chunk_pair[1]
    common_chroms = numpy.intersect1d(
        chunk1.vars_info[VAR_TABLE_CHROM_COL].values,
        chunk2.vars_info[VAR_TABLE_CHROM_COL].values,
    )
    if not common_chroms.size:
        return False

    chroms1 = chunk1.vars_info[VAR_TABLE_CHROM_COL]
    chroms2 = chunk2.vars_info[VAR_TABLE_CHROM_COL]
    poss1 = chunk1.vars_info[VAR_TABLE_POS_COL]
    poss2 = chunk2.vars_info[VAR_TABLE_POS_COL]
    for chrom in common_chroms:
        chrom_poss1 = poss1[chroms1 == chrom]
        chrom_poss2 = poss2[chroms2 == chrom]
        poss1_start = chrom_poss1.iloc[0]
        poss1_end = chrom_poss1.iloc[-1]
        poss2_start = chrom_poss2.iloc[0]
        poss2_end = chrom_poss2.iloc[-1]
        if poss1_start == poss2_start or poss1_end == poss2_end:
            return True
        dist = poss2_start - poss1_end
        if dist <= max_dist:
            return True

    return False


LDResult = namedtuple(
    "LDResult", ["r2", "chrom_var1", "pos_var1", "chrom_var2", "pos_var2", "dist_in_bp"]
)


def calc_pairwise_rogers_huff_r2(
    vars, max_dist: int | None = None, check_no_mafs_above: float | None = 0.95
):
    chunks = vars.iter_vars_chunks()
    chunk_pairs = itertools.combinations_with_replacement(chunks, 2)

    if max_dist is not None:
        chunks_are_close = partial(_chunks_are_close, max_dist=max_dist)
        chunk_pairs = filter(chunks_are_close, chunk_pairs)

    for chunk1, chunk2 in chunk_pairs:
        r2 = _calc_rogers_huff_r2(
            chunk1.gts.to_012(),
            chunk2.gts.to_012(),
            check_no_mafs_above=check_no_mafs_above,
        )

        poss1, poss2, chroms1, chroms2 = None, None, None, None
        # print(chunk1.vars_info)
        if chunk1.vars_info is not None and chunk2.vars_info is not None:
            try:
                poss1 = numpy.array(chunk1.vars_info[VAR_TABLE_POS_COL].values)
            except KeyError:
                pass
            try:
                poss2 = numpy.array(chunk2.vars_info[VAR_TABLE_POS_COL].values)
            except KeyError:
                pass
            try:
                chroms1 = numpy.array(chunk1.vars_info[VAR_TABLE_CHROM_COL].values)
            except KeyError:
                pass
            try:
                chroms2 = numpy.array(chunk2.vars_info[VAR_TABLE_CHROM_COL].values)
            except KeyError:
                pass

        both_chunks_are_same = chunk1 is chunk2
        for idx1 in range(chunk1.num_vars):
            for idx2 in range(chunk2.num_vars):
                if both_chunks_are_same and idx1 >= idx2:
                    continue
                if (
                    chroms1 is not None
                    and chroms2 is not None
                    and poss1 is not None
                    and poss2 is not None
                ):
                    pos1 = int(poss1[idx1])
                    pos2 = int(poss2[idx2])
                    chrom1 = chroms1[idx1]
                    chrom2 = chroms2[idx2]
                    dist = abs(pos1 - pos2) if chrom1 == chrom2 else None
                else:
                    pos1, pos2, chrom1, chrom2, dist = None, None, None, None, None

                if max_dist is not None:
                    if dist is None or dist > max_dist:
                        continue

                pair_r2 = float(r2[idx1, idx2])
                yield LDResult(pair_r2, chrom1, pos1, chrom2, pos2, dist)


# calc_ld_along_genome()
# calc_pairwise_rogers_huff_r2(max_dist)
#   returns a sparse matrix of (num_vars, num_vars)
#   return distance
# filter_vars_by_ld
