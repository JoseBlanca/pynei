import numpy
import pytest

import pynei.gt_counts
from pynei import (
    Variants,
    PerVarDistribs,
    PerVarStat,
    PolyVarsStats,
    StatsDistrib,
    calc_per_var_distribs,
    filter_by_missing_data,
)

NUM_SAMPLES = 30
SAMPLES = tuple(f"sample_{idx}" for idx in range(NUM_SAMPLES))
POPS = {"pop1": SAMPLES[:15], "pop2": SAMPLES[15:]}


def _create_vars(chunk_size=25):
    rng = numpy.random.default_rng(42)
    gts = rng.integers(0, 2, size=(100, NUM_SAMPLES, 2))
    variants = Variants.from_gt_array(gts, samples=SAMPLES)
    variants.desired_num_vars_per_chunk = chunk_size
    return variants


def test_all_the_stats_are_calculated_by_default():
    res = calc_per_var_distribs(_create_vars(), pops=POPS, min_num_samples=5)
    assert isinstance(res, PerVarDistribs)
    assert isinstance(res.obs_het, StatsDistrib)
    assert isinstance(res.maf, StatsDistrib)
    assert isinstance(res.exp_het, StatsDistrib)
    assert isinstance(res.poly_vars_ratio, PolyVarsStats)
    assert sorted(res.maf.mean.index) == ["pop1", "pop2"]


def test_only_the_stats_asked_for_are_calculated():
    res = calc_per_var_distribs(_create_vars(), stats={"maf"}, min_num_samples=5)
    assert isinstance(res.maf, StatsDistrib)
    assert res.obs_het is None
    assert res.exp_het is None
    assert res.poly_vars_ratio is None


def test_the_stats_can_be_given_as_strings_members_or_one_string():
    variants = _create_vars()
    by_str = calc_per_var_distribs(
        variants, stats={"maf", "exp_het"}, min_num_samples=5
    )
    by_member = calc_per_var_distribs(
        variants, stats={PerVarStat.MAF, PerVarStat.EXP_HET}, min_num_samples=5
    )
    assert numpy.allclose(by_str.maf.mean, by_member.maf.mean)
    assert numpy.allclose(by_str.exp_het.mean, by_member.exp_het.mean)
    assert by_str.obs_het is None and by_member.obs_het is None

    # one string is one stat, not its letters
    one = calc_per_var_distribs(variants, stats="maf", min_num_samples=5)
    assert isinstance(one.maf, StatsDistrib) and one.exp_het is None


def test_an_unknown_stat_is_refused():
    with pytest.raises(ValueError, match="not a valid PerVarStat"):
        calc_per_var_distribs(_create_vars(), stats={"mafs"})
    with pytest.raises(ValueError, match="At least one"):
        calc_per_var_distribs(_create_vars(), stats=[])


def test_the_stat_names_are_the_result_fields():
    for stat in PerVarStat:
        assert hasattr(PerVarDistribs(), str(stat))


def test_one_pass_gives_the_same_as_one_stat_at_a_time():
    variants = _create_vars()
    together = calc_per_var_distribs(
        variants, pops=POPS, min_num_samples=5, hist_kwargs={"num_bins": 10}
    )
    for stat in PerVarStat:
        alone = calc_per_var_distribs(
            variants,
            stats=stat,
            pops=POPS,
            min_num_samples=5,
            hist_kwargs={"num_bins": 10},
        )
        alone, in_pass = getattr(alone, stat), getattr(together, stat)
        if stat is PerVarStat.POLY_VARS_RATIO:
            assert numpy.allclose(alone.poly_ratio, in_pass.poly_ratio)
        else:
            assert numpy.allclose(alone.mean, in_pass.mean)
            assert numpy.all(alone.hist_counts.values == in_pass.hist_counts.values)


def test_the_allele_counts_are_calculated_once_per_chunk(monkeypatch):
    # maf, exp_het and poly_vars_ratio all need the allele counts, and the
    # cache shared by the calcs of a chunk has to spare the repeated work
    num_real_calcs = 0
    count_alleles = pynei.gt_counts._count_alleles_per_var

    def counting_count_alleles(chunk, calc_freqs, *args, cache=None, **kwargs):
        nonlocal num_real_calcs
        if cache is None:
            # the cached calls hand the work over to a call without cache
            num_real_calcs += 1
        return count_alleles(chunk, calc_freqs, *args, cache=cache, **kwargs)

    monkeypatch.setattr(
        pynei.gt_counts, "_count_alleles_per_var", counting_count_alleles
    )
    monkeypatch.setattr(
        pynei.diversity, "_count_alleles_per_var", counting_count_alleles
    )

    num_chunks = 4
    calc_per_var_distribs(_create_vars(chunk_size=25), min_num_samples=5)
    assert num_real_calcs == num_chunks


def test_threaded_run_gives_the_same_result():
    variants = _create_vars(chunk_size=10)
    serial = calc_per_var_distribs(variants, pops=POPS, min_num_samples=5)
    threaded = calc_per_var_distribs(
        variants, pops=POPS, min_num_samples=5, num_threads=2
    )
    assert numpy.allclose(serial.maf.mean, threaded.maf.mean)
    assert numpy.all(serial.maf.hist_counts.values == threaded.maf.hist_counts.values)
    assert numpy.allclose(serial.exp_het.mean, threaded.exp_het.mean)
    assert numpy.allclose(
        serial.poly_vars_ratio.num_poly, threaded.poly_vars_ratio.num_poly
    )


def test_the_conditional_kwargs_only_touch_their_stat():
    variants = _create_vars()
    biased = calc_per_var_distribs(
        variants, stats={"exp_het", "maf"}, min_num_samples=5, unbiased_exp_het=False
    )
    unbiased = calc_per_var_distribs(
        variants, stats={"exp_het", "maf"}, min_num_samples=5, unbiased_exp_het=True
    )
    assert not numpy.allclose(biased.exp_het.mean, unbiased.exp_het.mean)
    assert numpy.allclose(biased.maf.mean, unbiased.maf.mean)

    strict = calc_per_var_distribs(
        variants, stats="poly_vars_ratio", min_num_samples=5, poly_threshold=0.51
    )
    lax = calc_per_var_distribs(
        variants, stats="poly_vars_ratio", min_num_samples=5, poly_threshold=0.99
    )
    assert (strict.poly_vars_ratio.num_poly <= lax.poly_vars_ratio.num_poly).all()


def test_no_variants_is_an_error():
    variants = filter_by_missing_data(_create_vars(), max_allowed_missing_rate=-1)
    with pytest.raises(ValueError, match="no variants"):
        calc_per_var_distribs(variants)


def test_mapping_the_chunks_with_threads_keeps_every_chunk_in_order():
    # threaded_map_reduce 0.1.1 map() dropped results when the work reached
    # more than one thread, 0.1.2 is the first one that does not
    from pynei.pipeline import Pipeline

    pipeline = Pipeline(map_functs=[lambda chunk: chunk.num_vars])
    variants = _create_vars(chunk_size=5)
    serial = list(pipeline.map_chunks(variants))
    assert sum(serial) == 100
    for num_threads in (2, 4):
        assert list(pipeline.map_chunks(variants, num_threads=num_threads)) == serial
