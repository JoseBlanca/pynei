from dataclasses import dataclass
from functools import partial

import numpy
import pandas

from pynei.config import BinType
from pynei.pipeline import Pipeline


@dataclass(frozen=True)
class StatsDistrib:
    """The distribution, per pop, of a statistic calculated for every variant.

    The per variant values are not kept, because they do not fit in memory for
    a big dataset, what is kept is their mean and their histogram.
    """

    mean: pandas.Series
    "The mean of the statistic over all the variants, one value per pop"

    hist_bin_edges: numpy.ndarray
    "The edges of the histogram bins, there is one edge more than bins"

    hist_counts: pandas.DataFrame
    "How many variants fell in each histogram bin, one column per pop"


def _prepare_bins(
    hist_kwargs: dict | None = None,
    default_range: tuple[float, float] = (0, 1),
    default_num_bins: int = 40,
    default_bin_type: BinType = BinType.lineal,
):
    """It returns the edges of the histogram bins.

    hist_kwargs is the dict that the user gave, it is only read, never
    modified. It can have the keys range, num_bins and bin_type.
    """
    if hist_kwargs is None:
        hist_kwargs = {}

    hist_range = hist_kwargs.get("range", default_range)
    num_bins = hist_kwargs.get("num_bins", default_num_bins)
    # BinType is a StrEnum, so this takes both the strings and the members, and
    # it raises for anything else
    bin_type = BinType(hist_kwargs.get("bin_type", default_bin_type))

    if bin_type is BinType.lineal:
        bins = numpy.linspace(hist_range[0], hist_range[1], num_bins + 1)
    else:
        if hist_range[0] <= 0:
            raise ValueError(
                f"range[0] should be positive for logarithmic bins, but it is {hist_range[0]}"
            )
        # logspace takes the exponents, not the limits of the range
        bins = numpy.logspace(
            numpy.log10(hist_range[0]), numpy.log10(hist_range[1]), num_bins + 1
        )
    return bins


def _collect_stats_from_pop_dframes(
    accumulated_result, next_result: pandas.DataFrame, hist_bins_edges: numpy.array
):
    if accumulated_result is None:
        accumulated_result = {
            "sum_per_pop": pandas.Series(
                numpy.zeros((next_result.shape[1]), dtype=int),
                index=next_result.columns,
            ),
            "total_num_rows": pandas.Series(
                numpy.zeros((next_result.shape[1]), dtype=int),
                index=next_result.columns,
            ),
            "hist_counts": None,
        }

    accumulated_result["sum_per_pop"] += next_result.sum(axis=0)
    accumulated_result["total_num_rows"] += next_result.shape[
        0
    ] - next_result.isna().sum(axis=0)

    this_counts = {}
    for pop, pop_stats in next_result.items():
        this_counts[pop] = numpy.histogram(pop_stats, bins=hist_bins_edges)[0]
    this_counts = pandas.DataFrame(this_counts)

    if accumulated_result["hist_counts"] is None:
        accumulated_result["hist_counts"] = this_counts
    else:
        accumulated_result["hist_counts"] += this_counts

    return accumulated_result


def _calc_per_var_distrib(
    variants,
    calc_stats_for_chunk,
    get_stats_for_chunk_result,
    hist_kwargs=None,
    default_hist_range: tuple[float, float] = (0, 1),
):
    hist_bins_edges = _prepare_bins(hist_kwargs, default_range=default_hist_range)

    collect_stats_from_pop_dframes = partial(
        _collect_stats_from_pop_dframes, hist_bins_edges=hist_bins_edges
    )

    pipeline = Pipeline(
        map_functs=[
            calc_stats_for_chunk,
            get_stats_for_chunk_result,
        ],
        reduce_funct=collect_stats_from_pop_dframes,
    )
    accumulated_result = pipeline.map_and_reduce(variants)

    mean = accumulated_result["sum_per_pop"] / accumulated_result["total_num_rows"]
    return StatsDistrib(
        mean=mean,
        hist_bin_edges=hist_bins_edges,
        hist_counts=accumulated_result["hist_counts"],
    )
