import numpy
import pytest

from pynei import (
    Variants,
    calc_obs_het_stats_per_var,
    calc_major_allele_stats_per_var,
    calc_exp_het_stats_per_var,
)
from pynei.config import BinType, LINEAL, LOGARITHMIC
from pynei.utils_stats import _prepare_bins


def _create_vars():
    numpy.random.seed(42)
    gts = numpy.random.randint(0, 2, size=(20, 25, 2))
    return Variants.from_gt_array(gts, samples=[f"s{idx}" for idx in range(25)])


@pytest.mark.parametrize(
    "calc_stats",
    [
        calc_obs_het_stats_per_var,
        calc_major_allele_stats_per_var,
        calc_exp_het_stats_per_var,
    ],
)
def test_hist_kwargs_is_not_modified(calc_stats):
    variants = _create_vars()

    hist_kwargs = {}
    calc_stats(variants, hist_kwargs=hist_kwargs)
    assert hist_kwargs == {}

    hist_kwargs = {"num_bins": 4}
    calc_stats(variants, hist_kwargs=hist_kwargs)
    assert hist_kwargs == {"num_bins": 4}


def test_hist_range_can_be_given():
    variants = _create_vars()
    res = calc_obs_het_stats_per_var(
        variants, hist_kwargs={"num_bins": 2, "range": (0, 0.5)}
    )
    assert numpy.allclose(res.hist_bin_edges, [0, 0.25, 0.5])


def test_logarithmic_bins_span_the_given_range():
    bins = _prepare_bins({"range": (0.01, 100), "num_bins": 4, "bin_type": LOGARITHMIC})
    assert numpy.allclose(bins, [0.01, 0.1, 1, 10, 100])


def test_bins_take_the_bin_type_as_a_string_or_as_a_member():
    lineal = _prepare_bins({"range": (0, 1), "num_bins": 2, "bin_type": LINEAL})
    assert numpy.allclose(lineal, [0, 0.5, 1])
    assert numpy.allclose(
        _prepare_bins({"range": (0, 1), "num_bins": 2, "bin_type": BinType.lineal}),
        lineal,
    )
    assert numpy.allclose(_prepare_bins({"range": (0, 1), "num_bins": 2}), lineal)

    log = _prepare_bins({"range": (0.1, 10), "num_bins": 2, "bin_type": LOGARITHMIC})
    assert numpy.allclose(
        _prepare_bins(
            {"range": (0.1, 10), "num_bins": 2, "bin_type": BinType.logarithmic}
        ),
        log,
    )


def test_an_unknown_bin_type_is_refused():
    with pytest.raises(ValueError):
        _prepare_bins({"range": (0, 1), "bin_type": "quadratic"})


def test_logarithmic_bins_refuse_a_non_positive_range():
    for bad_range in ((0, 10), (-1, 10)):
        with pytest.raises(ValueError, match="should be positive"):
            _prepare_bins({"range": bad_range, "bin_type": LOGARITHMIC})
