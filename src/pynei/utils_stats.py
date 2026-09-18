from dataclasses import dataclass

import numpy
import pandas

from pynei.config import BinType


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


def _summarize_pop_dframe(stat_per_var: pandas.DataFrame, hist_bin_edges):
    """The contribution of one chunk to a StatsDistrib.

    stat_per_var has one row per variant and one column per pop.
    """
    hist_counts = {
        pop: numpy.histogram(pop_stats, bins=hist_bin_edges)[0]
        for pop, pop_stats in stat_per_var.items()
    }
    return {
        "sum_per_pop": stat_per_var.sum(axis=0),
        "num_vars_with_data": stat_per_var.notna().sum(axis=0),
        "hist_counts": pandas.DataFrame(hist_counts),
    }


def _add_distrib_contributions(accumulated, contribution):
    return {
        "sum_per_pop": accumulated["sum_per_pop"] + contribution["sum_per_pop"],
        "num_vars_with_data": accumulated["num_vars_with_data"]
        + contribution["num_vars_with_data"],
        "hist_counts": accumulated["hist_counts"] + contribution["hist_counts"],
    }


def _finish_distrib(accumulated, hist_bin_edges) -> StatsDistrib:
    return StatsDistrib(
        mean=accumulated["sum_per_pop"] / accumulated["num_vars_with_data"],
        hist_bin_edges=hist_bin_edges,
        hist_counts=accumulated["hist_counts"],
    )
