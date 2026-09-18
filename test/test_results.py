import dataclasses

import numpy
import pytest

import pynei
from pynei import (
    Variants,
    FilteringStats,
    PCAResult,
    PCoAResult,
    PolyVarsStats,
    R2Matrix,
    StatsDistrib,
    calc_exp_het_per_var_distrib,
    calc_maf_per_var_distrib,
    calc_obs_het_per_var_distrib,
    calc_poly_vars_ratio,
    calc_rogers_huff_r2_matrix,
    do_pca_from_variants,
    do_pcoa_from_variants,
    filter_by_maf,
    gather_filtering_stats,
)

RESULT_CLASSES = [
    FilteringStats,
    PCAResult,
    PCoAResult,
    PolyVarsStats,
    R2Matrix,
    StatsDistrib,
]


def _create_vars():
    numpy.random.seed(42)
    gt_array = numpy.random.randint(0, 2, size=(20, 25, 2))
    return Variants.from_gt_array(
        gt_array, samples=[f"sample_{idx}" for idx in range(25)]
    )


@pytest.mark.parametrize("result_class", RESULT_CLASSES)
def test_the_results_are_frozen_dataclasses_with_documented_fields(result_class):
    assert dataclasses.is_dataclass(result_class)
    assert result_class.__doc__
    fields = dataclasses.fields(result_class)
    assert fields
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance = result_class(**{field.name: None for field in fields})
        setattr(instance, fields[0].name, None)


@pytest.mark.parametrize(
    "calc_stats",
    [
        calc_obs_het_per_var_distrib,
        calc_maf_per_var_distrib,
        calc_exp_het_per_var_distrib,
    ],
)
def test_the_per_var_distribs(calc_stats):
    res = calc_stats(_create_vars(), hist_kwargs={"num_bins": 4})
    assert isinstance(res, StatsDistrib)
    assert res.mean.shape == (1,)
    assert res.hist_bin_edges.shape == (5,)
    assert res.hist_counts.shape == (4, 1)
    # a typo is an error now, it used to be a KeyError only if it was a lookup
    with pytest.raises(AttributeError):
        res.hist_count


def test_poly_vars_ratio_gives_a_poly_vars_stats():
    res = calc_poly_vars_ratio(_create_vars(), min_num_samples=1)
    assert isinstance(res, PolyVarsStats)
    assert res.poly_ratio.shape == (1,)


def test_the_pca_and_the_pcoa_give_their_own_results():
    res = do_pca_from_variants(_create_vars())
    assert isinstance(res, PCAResult)
    assert res.projections.shape[0] == 25
    assert res.explained_variance_percent.sum() == pytest.approx(100)
    assert res.princomps.shape[0] == res.projections.shape[1]

    res = do_pcoa_from_variants(_create_vars())
    assert isinstance(res, PCoAResult)
    assert res.projections.shape[0] == 25
    assert not hasattr(res, "princomps")


def test_the_r2_matrix_result():
    res = calc_rogers_huff_r2_matrix(_create_vars(), check_no_mafs_above=None)
    assert isinstance(res, R2Matrix)
    assert res.r2.shape == (20, 20)
    # these variants carry no chrom and no pos
    assert res.dists_in_bp is None


def test_the_filtering_stats():
    variants = filter_by_maf(_create_vars(), max_allowed_maf=1)
    list(variants.iter_vars_chunks())
    stats = gather_filtering_stats(variants)
    assert stats == {"maf": FilteringStats(vars_processed=20, vars_kept=20)}
    assert isinstance(stats["maf"].vars_kept, int)


def test_the_result_types_are_exported():
    for result_class in RESULT_CLASSES:
        assert result_class.__name__ in pynei.__all__
