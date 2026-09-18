import itertools
from dataclasses import dataclass
from functools import partial
from collections import namedtuple
from enum import Enum

import numpy
import pandas
import more_itertools

from pynei.config import VAR_TABLE_CHROM_COL, VAR_TABLE_POS_COL, DEF_POP_NAME
from pynei.var_filters import filter_by_maf, filter_samples
from pynei.utils_pop import Pops, _calc_pops_idxs
from .ld_calc import _calc_rogers_huff_r2


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


@dataclass(frozen=True)
class R2Matrix:
    """The Rogers Huff r2 between every pair of variants."""

    r2: numpy.ndarray
    "A square vars x vars matrix with the r2 of every pair of variants"

    dists_in_bp: numpy.ndarray | None = None
    """A square vars x vars matrix with the distance between every pair of
    variants, nan when they are in different chroms. It is None when the
    variants carry no chrom and pos."""


def calc_rogers_huff_r2_matrix(
    variants, max_dist: int | None = None, check_no_mafs_above: float | None = 0.95
):
    # This function is faster than calc_pairwise_rogers_huff_r2,
    # but it uses much more memory
    chunks = list(variants.iter_vars_chunks())
    tot_num_vars = sum(chunk.num_vars for chunk in chunks)
    r2 = numpy.full((tot_num_vars, tot_num_vars), numpy.nan)
    dists = None
    row_start = 0
    for chunk1 in chunks:
        col_start = 0
        for chunk2 in chunks:
            if max_dist:
                if not _chunks_are_close((chunk1, chunk2), max_dist):
                    continue
            row_end = row_start + chunk1.num_vars
            col_end = col_start + chunk2.num_vars

            this_r2 = _calc_rogers_huff_r2(
                chunk1.gts.to_012(),
                chunk2.gts.to_012(),
                check_no_mafs_above=check_no_mafs_above,
            )
            r2[row_start:row_end, col_start:col_end] = this_r2

            chroms1, poss1, chroms2, poss2 = None, None, None, None
            if chunk1.vars_info is not None:
                try:
                    chroms1 = chunk1.vars_info[VAR_TABLE_CHROM_COL]
                except KeyError:
                    pass
                try:
                    poss1 = chunk1.vars_info[VAR_TABLE_POS_COL].to_numpy()
                except KeyError:
                    pass
                try:
                    chroms2 = chunk2.vars_info[VAR_TABLE_CHROM_COL]
                except KeyError:
                    pass
                try:
                    poss2 = chunk2.vars_info[VAR_TABLE_POS_COL].to_numpy()
                except KeyError:
                    pass
                if not any(
                    [
                        chroms1 is None,
                        chroms2 is None,
                        poss1 is None,
                        poss2 is None,
                    ]
                ):
                    mat1 = numpy.repeat(poss1, chunk2.num_vars).reshape(
                        (chunk1.num_vars, chunk2.num_vars)
                    )
                    mat2 = numpy.tile(poss2, chunk1.num_vars).reshape(
                        (chunk1.num_vars, chunk2.num_vars)
                    )
                    this_dists = numpy.abs(mat1 - mat2).astype(float)

                    num_chroms1 = chroms1.size
                    chroms = pandas.concat([chroms1, chroms2])
                    chroms = pandas.factorize(chroms)[0]
                    chroms1 = chroms[:num_chroms1]
                    chroms2 = chroms[num_chroms1:]
                    mat1 = numpy.repeat(chroms1, chunk2.num_vars).reshape(
                        (chunk1.num_vars, chunk2.num_vars)
                    )
                    mat2 = numpy.tile(chroms2, chunk1.num_vars).reshape(
                        (chunk1.num_vars, chunk2.num_vars)
                    )
                    is_different_chrom = mat1 != mat2
                    this_dists[is_different_chrom] = numpy.nan

                    if dists is None:
                        dists = numpy.full((tot_num_vars, tot_num_vars), numpy.nan)
                    dists[row_start:row_end, col_start:col_end] = this_dists

            col_start = col_end
        row_start = row_end

    return R2Matrix(r2=r2, dists_in_bp=dists)


LDResult = namedtuple(
    "LDResult", ["r2", "chrom_var1", "pos_var1", "chrom_var2", "pos_var2", "dist_in_bp"]
)


def calc_pairwise_rogers_huff_r2(
    variants, max_dist: int | None = None, check_no_mafs_above: float | None = 0.95
):
    # This is the slower alternative, calc_rogers_huff_r2_matrix is much faster,
    # but if you have many variants and the calculation does not fit in memory, use this one
    chunks = variants.iter_vars_chunks()
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


class LDCalcMethod(Enum):
    GENERATOR = "generator"
    MATRIX = "matrix"


def get_ld_and_dist_for_pops(
    variants,
    pops: Pops | None = None,
    max_dist: int | None = None,
    min_dist: int | None = 1,
    max_allowed_maf=0.95,
    method=LDCalcMethod.GENERATOR,
    max_num_measures_to_keep=10000,
):
    if pops is None:
        # one pop with every sample, there is nothing to filter
        pops = {DEF_POP_NAME: None}
    else:
        # it is only called to check that the pops are right, the idxs are not
        # used here, filter_samples works with the sample names
        _calc_pops_idxs(pops, variants.samples)

    ld_per_pop = {}
    for pop_name, pop_samples in pops.items():
        pop_vars = variants
        if pop_samples is not None:
            pop_vars = filter_samples(pop_vars, pop_samples)
        pop_vars = filter_by_maf(pop_vars, max_allowed_maf=max_allowed_maf)
        if method == LDCalcMethod.GENERATOR:
            lds_and_dists = (
                (res.r2, res.dist_in_bp)
                for res in calc_pairwise_rogers_huff_r2(
                    pop_vars, max_dist=max_dist, check_no_mafs_above=None
                )
                if res.dist_in_bp is not None
            )
        elif method == LDCalcMethod.MATRIX:
            res = calc_rogers_huff_r2_matrix(
                pop_vars, max_dist=max_dist, check_no_mafs_above=None
            )
            r2 = res.r2.flat
            dists = res.dists_in_bp.flat
            mask = ~numpy.isnan(dists)
            r2 = r2[mask]
            dists = dists[mask]
            lds_and_dists = [(float(r2), float(ld)) for r2, ld in zip(r2, dists)]

        if min_dist:
            lds_and_dists = filter(lambda x: x[1] > min_dist, lds_and_dists)
        try:
            lds_and_dists = more_itertools.sample(
                lds_and_dists, k=max_num_measures_to_keep, strict=False
            )
        except TypeError:
            # old versions of more-itertools.sample seem to lack the strict argument
            lds_and_dists = more_itertools.sample(
                lds_and_dists,
                k=max_num_measures_to_keep,
            )
        ld_per_pop[pop_name] = lds_and_dists
    return ld_per_pop


# calc_ld_along_genome()
# filter_vars_by_ld
