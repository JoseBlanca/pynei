from functools import partial
import itertools
from typing import Sequence

import numpy

from pynei.variants import Variants, VariantsChunk
from pynei.gt_counts import (
    _calc_gt_is_missing,
    _calc_maf_per_var,
    _calc_obs_het_per_var,
)


class _FilterChunkIterFactory:
    def __init__(self, in_vars, filter_funct):
        self.in_vars = in_vars
        self._chunks = in_vars.iter_vars_chunks()
        self.filter_funct = filter_funct
        self.num_vars_processed = 0
        self.num_vars_kept = 0
        self._metadata = None

    def _get_metadata(self):
        if self._metadata is not None:
            return self._metadata.copy()

        try:
            first_chunk = next(self.iter_vars_chunks())
        except StopIteration:
            raise RuntimeError("No variations to get the data from")

        self._chunks = itertools.chain([first_chunk], self._chunks)

        self._metadata = {
            "samples": first_chunk.gts.samples,
            "num_samples": first_chunk.num_samples,
            "ploidy": first_chunk.gts.ploidy,
        }
        return self._metadata.copy()

    def iter_vars_chunks(self):
        for chunk in self._chunks:
            if self._metadata is None:
                self._metadata = {
                    "samples": chunk.gts.samples,
                    "num_samples": chunk.num_samples,
                    "ploidy": chunk.gts.ploidy,
                }
            filtered_chunk, num_vars_kept = self.filter_funct(chunk)
            self.num_vars_processed += chunk.num_vars
            self.num_vars_kept += num_vars_kept
            yield filtered_chunk


class _MissingFilterIterFactory(_FilterChunkIterFactory):
    kind = "missing_data"


def gather_filtering_stats(vars: Variants, stats=None):
    if stats is None:
        stats = {}  # stats by filter kind
    chunk_factory = vars._vars_chunks_iter_factory
    if isinstance(chunk_factory, _FilterChunkIterFactory):
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


class _MafFilterIterFactory(_FilterChunkIterFactory):
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


class _ObsHetFilterIterFactory(_FilterChunkIterFactory):
    kind = "obs_het"


def _filter_chunk_by_obs_het(chunk, max_allowed_obs_het):
    obs_hets = _calc_obs_het_per_var(chunk, pops={"pop": slice(None, None)})[
        "obs_het_per_var"
    ]["pop"]
    mask = obs_hets <= max_allowed_obs_het
    num_vars_kept = mask.sum()
    chunk = chunk.get_vars(mask)
    return chunk, num_vars_kept


def filter_by_obs_het(vars: Variants, max_allowed_obs_het: float):
    filter_chunk = partial(
        _filter_chunk_by_obs_het, max_allowed_obs_het=max_allowed_obs_het
    )
    chunk_factory = _ObsHetFilterIterFactory(vars, filter_chunk)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


def _filter_samples(chunk, sample_idxs):
    gts = chunk.gts.filter_samples_with_idxs(sample_idxs)
    chunk = VariantsChunk(gts, vars_info=chunk.vars_info, alleles=chunk.alleles)
    return chunk, chunk.num_vars


def filter_samples(vars, samples: Sequence[str] | Sequence[int] | slice) -> Variants:
    orig_samples = vars.samples
    if isinstance(samples, slice):
        samples = orig_samples[samples]
    sample_idxs = numpy.where(numpy.isin(orig_samples, samples))[0]

    filter_samples = partial(_filter_samples, sample_idxs=sample_idxs)
    chunk_factory = _MissingFilterIterFactory(vars, filter_samples)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


# TODO
# var QUAL
