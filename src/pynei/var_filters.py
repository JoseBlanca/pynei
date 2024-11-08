from functools import partial

import numpy

from pynei.variants import Variants
from pynei.gt_counts import _calc_gt_is_missing


class _FilterIterFactory:
    def __init__(self, in_vars, filter_funct):
        self.in_vars = in_vars
        self.filter_funct = filter_funct

    def iter_vars_chunks(self):
        filter_funct = self.filter_funct
        return (filter_funct(chunk) for chunk in self.in_vars.iter_vars_chunks())


def _filter_chunk_by_missing(chunk, max_missing_rate):
    num_missing_per_var = numpy.sum(_calc_gt_is_missing(chunk)["gt_is_missing"], axis=1)
    missing_rate_per_var = num_missing_per_var / chunk.num_samples
    chunk = chunk.get_vars(missing_rate_per_var <= max_missing_rate)
    return chunk


def filter_by_missing_data(vars: Variants, max_missing_rate: float = 0.0) -> Variants:
    filter_chunk_by_missing = partial(
        _filter_chunk_by_missing, max_missing_rate=max_missing_rate
    )
    chunk_factory = _FilterIterFactory(vars, filter_chunk_by_missing)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


# TODO
# poly_rate
# obs_het
# var QUAL
