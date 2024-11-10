from functools import partial

import numpy

from pynei.variants import Variants
from pynei.gt_counts import _calc_gt_is_missing, _calc_maf_per_var


class _FilterIterFactory:
    def __init__(self, in_vars, filter_funct):
        self.in_vars = in_vars
        self.filter_funct = filter_funct
        self.num_vars_processed = 0
        self.num_vars_kept = 0

    def iter_vars_chunks(self):
        filter_funct = self.filter_funct
        for chunk in self.in_vars.iter_vars_chunks():
            filtered_chunk, num_vars_kept = filter_funct(chunk)
            self.num_vars_processed += chunk.num_vars
            self.num_vars_kept += num_vars_kept
            yield filtered_chunk


class _MissingFilterIterFactory(_FilterIterFactory):
    kind = "missing_data"


def gather_filtering_stats(vars: Variants, stats=None):
    if stats is None:
        stats = {}  # stats by filter kind
    chunk_factory = vars._vars_chunks_iter_factory
    if isinstance(chunk_factory, _FilterIterFactory):
        filter_kind = chunk_factory.kind
        if filter_kind not in stats:
            stats[filter_kind] = {"vars_processed": 0, "vars_kept": 0}
        stats[filter_kind]["vars_processed"] += chunk_factory.num_vars_processed
        stats[filter_kind]["vars_kept"] += chunk_factory.num_vars_kept
    if hasattr(chunk_factory, "in_vars"):
        gather_filtering_stats(chunk_factory.in_vars, stats)

    for filtering_stats in stats.values():
        if "vars_kept" in filtering_stats and hasattr(
            filtering_stats["vars_kept"], "item"
        ):
            # this is to convert from np.int64 to native python int
            filtering_stats["vars_kept"] = filtering_stats["vars_kept"].item()
    return stats


def _filter_chunk_by_missing(chunk, max_missing_rate):
    num_missing_per_var = numpy.sum(_calc_gt_is_missing(chunk)["gt_is_missing"], axis=1)
    missing_rate_per_var = num_missing_per_var / chunk.num_samples
    mask = missing_rate_per_var <= max_missing_rate
    num_vars_kept = mask.sum()
    chunk = chunk.get_vars(mask)
    return chunk, num_vars_kept


def filter_by_missing_data(
    vars: Variants, max_allowed_missing_rate: float = 0.0
) -> Variants:
    filter_chunk_by_missing = partial(
        _filter_chunk_by_missing, max_missing_rate=max_allowed_missing_rate
    )
    chunk_factory = _MissingFilterIterFactory(vars, filter_chunk_by_missing)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


class _MafFilterIterFactory(_FilterIterFactory):
    kind = "maf"


def _filter_chunk_by_maf(chunk, max_allowed_maf):
    mafs = _calc_maf_per_var(chunk, pops={"pop": slice(None, None)}, min_num_samples=0)[
        "major_allele_freqs_per_var"
    ]["pop"]
    mask = mafs <= max_allowed_maf
    num_vars_kept = mask.sum()
    chunk = chunk.get_vars(mask)
    return chunk, num_vars_kept


def filter_by_maf(vars: Variants, max_allowed_maf) -> Variants:
    filter_chunk = partial(_filter_chunk_by_maf, max_allowed_maf=max_allowed_maf)
    chunk_factory = _MafFilterIterFactory(vars, filter_chunk)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


# TODO
# obs_het
# var QUAL
