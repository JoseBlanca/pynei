import numpy
import pytest

from pynei import (
    Variants,
    calc_obs_het_stats_per_var,
    calc_major_allele_stats_per_var,
    calc_exp_het_stats_per_var,
    calc_poly_vars_ratio_per_var,
    calc_jost_dest_pop_dists,
    get_ld_and_dist_for_pops,
)
from pynei.utils_pop import _calc_pops_idxs

SAMPLES = tuple(f"sample_{idx}" for idx in range(6))
POPS = {"pop1": SAMPLES[:3], "pop2": SAMPLES[3:]}


def _create_vars():
    numpy.random.seed(42)
    gts = numpy.random.randint(0, 2, size=(10, 6, 2))
    return Variants.from_gt_array(gts, samples=SAMPLES)


def test_pops_idxs_from_sample_names():
    assert _calc_pops_idxs(POPS, SAMPLES) == {"pop1": [0, 1, 2], "pop2": [3, 4, 5]}
    # the order that the user gave is kept
    assert _calc_pops_idxs({"p": ("sample_3", "sample_0")}, SAMPLES) == {"p": [3, 0]}
    # no pops means one pop with every sample
    assert _calc_pops_idxs(None, SAMPLES) == {"pop": slice(None, None)}


def test_a_sample_that_is_not_in_the_variants_is_reported():
    with pytest.raises(ValueError, match="not in the variants"):
        _calc_pops_idxs({"pop1": ["sample_0", "nope"]}, SAMPLES)


def test_pops_have_to_be_sequences_of_sample_names():
    # a slice used to work only in get_ld_and_dist_for_pops
    with pytest.raises(ValueError, match="sequence of sample names"):
        _calc_pops_idxs({"pop1": slice(3)}, SAMPLES)
    # a single sample name is not a sequence of sample names
    with pytest.raises(ValueError, match="sequence of sample names"):
        _calc_pops_idxs({"pop1": "sample_0"}, SAMPLES)


def test_pops_need_samples_in_the_variants():
    gts = numpy.random.randint(0, 2, size=(10, 6, 2))
    variants = Variants.from_gt_array(gts)
    with pytest.raises(ValueError, match="should have samples"):
        calc_obs_het_stats_per_var(variants, pops=POPS)


@pytest.mark.parametrize(
    "calc_stats",
    [
        calc_obs_het_stats_per_var,
        calc_major_allele_stats_per_var,
        calc_exp_het_stats_per_var,
        calc_poly_vars_ratio_per_var,
    ],
)
def test_every_stat_takes_the_same_pops(calc_stats):
    res = calc_stats(_create_vars(), pops=POPS)
    per_pop = res["mean"] if "mean" in res else res["num_poly"]
    assert sorted(per_pop.index) == ["pop1", "pop2"]

    with pytest.raises(ValueError, match="not in the variants"):
        calc_stats(_create_vars(), pops={"pop1": ["nope"]})


def test_jost_dest_takes_the_same_pops():
    dists = calc_jost_dest_pop_dists(_create_vars(), pops=POPS, min_num_samples=1)
    assert sorted(dists.names) == ["pop1", "pop2"]

    with pytest.raises(ValueError, match="not in the variants"):
        calc_jost_dest_pop_dists(_create_vars(), pops={"p1": ["nope"], "p2": ["x"]})


def test_ld_for_pops_takes_the_same_pops():
    res = get_ld_and_dist_for_pops(_create_vars(), pops=POPS, max_allowed_maf=1)
    assert sorted(res.keys()) == ["pop1", "pop2"]

    with pytest.raises(ValueError, match="not in the variants"):
        get_ld_and_dist_for_pops(_create_vars(), pops={"pop1": ["nope"]})
