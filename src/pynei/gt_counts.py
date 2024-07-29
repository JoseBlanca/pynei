from functools import partial

import numpy
import pandas

from pynei.pipeline import Pipeline
from pynei.config import MISSING_ALLELE, BinType, LINEAL, LOGARITHMIC, DEF_POP_NAME
from pynei.variants import Variants


def _calc_gt_is_missing(chunk, partial_res=None):
    res = {} if partial_res is None else partial_res
    if "gt_is_missing" in res:
        return res

    allele_is_missing = chunk.gts.gt_array == MISSING_ALLELE
    res["gt_is_missing"] = numpy.any(allele_is_missing, axis=2)
    return res


def _calc_gt_is_het(chunk, partial_res=None):
    res = {} if partial_res is None else partial_res
    if "gt_is_het" in res:
        return res

    res = _calc_gt_is_missing(chunk, partial_res=res)
    gt_is_missing = res["gt_is_missing"]

    gt_array = chunk.gts.gt_array
    gt_is_het = numpy.logical_not(
        numpy.all(gt_array == gt_array[:, :, 0][:, :, numpy.newaxis], axis=2)
    )
    res["gt_is_het"] = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))
    return res


def _calc_obs_het_per_var(chunk, pops):
    res = _calc_gt_is_het(chunk)
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
    return {"obs_het_per_var": obs_het_per_var}


def _calc_pops_idxs(pops: dict[list[str]] | None, samples):
    if pops is None:
        pops_idxs = {DEF_POP_NAME: slice(None, None)}
    else:
        if samples is None:
            raise ValueError("Variants should have samples defined if pops is not None")
        samples_idx = {sample: idx for idx, sample in enumerate(samples)}
        pops_idxs = {}
        for pop_id, pop_samples in pops.items():
            pops_idxs[pop_id] = [samples_idx[sample] for sample in pop_samples]
    return pops_idxs


def _prepare_bins(
    hist_kwargs: dict,
    range=tuple[int, int],
    default_num_bins=40,
    default_bin_type: BinType = LINEAL,
):
    num_bins = hist_kwargs.get("num_bins", default_num_bins)
    bin_type = hist_kwargs.get("bin_type", default_bin_type)

    if bin_type == LINEAL:
        bins = numpy.linspace(range[0], range[1], num_bins + 1)
    elif bin_type == LOGARITHMIC:
        if range[0] == 0:
            raise ValueError("range[0] cannot be zero for logarithmic bins")
        bins = numpy.logspace(range[0], range[1], num_bins + 1)
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


def _calc_stats_per_var(
    variants,
    calc_stats_for_chunk,
    get_stats_for_chunk_result,
    hist_kwargs=None,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_bins_edges = _prepare_bins(hist_kwargs, range=hist_kwargs["range"])

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
    return {
        "mean": mean,
        "hist_bin_edges": hist_bins_edges,
        "hist_counts": accumulated_result["hist_counts"],
    }


def calc_obs_het_stats_per_var(
    variants: Variants,
    pops: list[str] | None = None,
    hist_kwargs=None,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["range"] = hist_kwargs.get("range", (0, 1))

    pops = _calc_pops_idxs(pops, variants.samples)

    return _calc_stats_per_var(
        variants=variants,
        calc_stats_for_chunk=partial(_calc_obs_het_per_var, pops=pops),
        get_stats_for_chunk_result=lambda x: x["obs_het_per_var"],
        hist_kwargs=hist_kwargs,
    )
