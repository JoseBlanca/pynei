import pytest
import numpy

from pynei.variants import Variants
from pynei.var_filters import (
    filter_by_missing_data,
    filter_by_maf,
    filter_by_obs_het,
    gather_filtering_stats,
    FilteringStats,
    filter_samples,
    filter_by_ld_and_maf,
)
from pynei import calc_per_var_distribs
from .var_generators import _FromGtListChunkIterFactory


def test_filter_missing():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_missing_data(orig_vars, max_allowed_missing_rate=0)
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, False], ...] == filtered_gts)

    variants = filter_by_missing_data(orig_vars, max_allowed_missing_rate=0.5)
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, True, False], ...] == filtered_gts)

    variants = filter_by_missing_data(orig_vars, max_allowed_missing_rate=-0.1)
    with pytest.raises(StopIteration):
        next(variants.iter_vars_chunks())


def test_filter_mafs():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_maf(orig_vars, max_allowed_maf=0.9)
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, True, False], ...] == filtered_gts)

    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_maf(orig_vars, max_allowed_maf=0.7)
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[False, True, False], ...] == filtered_gts)

    stats = gather_filtering_stats(variants)
    assert stats == {"maf": FilteringStats(vars_processed=3, vars_kept=1)}


def test_filter_obs_het():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_obs_het(orig_vars, max_allowed_obs_het=1.5 / 5.0)
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, True], ...] == filtered_gts)

    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_obs_het(orig_vars, max_allowed_obs_het=0)
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[False, False, True], ...] == filtered_gts)

    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_obs_het(orig_vars, max_allowed_obs_het=4 / 5)
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, True, True], ...] == filtered_gts)


def test_metadata():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    # before filtering
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_obs_het(orig_vars, max_allowed_obs_het=1.5 / 5.0)
    assert variants.num_samples == 5
    assert variants.ploidy == 2
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, True], ...] == filtered_gts)

    # after filtering
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_obs_het(orig_vars, max_allowed_obs_het=1.5 / 5.0)
    filtered_gts = numpy.ma.getdata(next(variants.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, True], ...] == filtered_gts)
    assert variants.num_samples == 5
    assert variants.ploidy == 2


def test_filter_samples():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_samples(orig_vars, samples=[0, 1, 2])
    assert numpy.all(next(variants.iter_vars_chunks()).gts.gt_values == gts[:, :3, :])

    variants = filter_samples(orig_vars, samples=slice(3))
    assert numpy.all(next(variants.iter_vars_chunks()).gts.gt_values == gts[:, :3, :])


def test_filter_ld():
    gts = numpy.array(
        [
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    with pytest.raises(StopIteration):
        next(variants.iter_vars_chunks())

    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [1, 0], [2, 2], [1, 0], [0, 0]],
            [[1, 1], [2, 2], [0, 1], [0, 0], [1, 2]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(variants.iter_vars_chunks())
    assert numpy.all(gts[[True, True, True], :] == chunk.gts.gt_values)

    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 0], [2, 0], [1, 0], [0, 0]],
            [[1, 0], [0, 2], [0, 1], [0, 0], [2, 2]],
            [[1, 0], [0, 2], [0, 1], [0, 0], [2, 2]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(variants.iter_vars_chunks())
    assert numpy.all(gts[[True, False, False, True, False], :] == chunk.gts.gt_values)

    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(variants.iter_vars_chunks())
    assert numpy.all(gts[[True, False, False], :] == chunk.gts.gt_values)

    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(variants.iter_vars_chunks())
    assert numpy.all(gts[[True, False, False], :] == chunk.gts.gt_values)

    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 0], [2, 0], [1, 0], [0, 0]],
            [[1, 0], [0, 2], [0, 1], [0, 0], [2, 2]],
            [[1, 0], [0, 2], [0, 1], [0, 0], [2, 2]],
        ]
    )
    for len_chunk in range(1, 5):
        orig_vars = Variants(
            _FromGtListChunkIterFactory(gts=[gts]),
            desired_num_vars_per_chunk=len_chunk,
        )
        variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
        filtered_gts = numpy.vstack(
            [chunk.gts.gt_values for chunk in variants.iter_vars_chunks()]
        )
        assert numpy.all(gts[[True, False, False, True, False], :] == filtered_gts)

        stats = gather_filtering_stats(variants)
        assert stats == {"ld_and_maf": FilteringStats(vars_processed=5, vars_kept=2)}


def test_get_metadata():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [1, 0], [2, 2], [1, 0], [0, 0]],
            [[1, 1], [2, 2], [0, 1], [0, 0], [1, 2]],
        ]
    )
    samples = [0, 1, 2, 3, 4]
    orig_vars = Variants.from_gt_array(gts, samples=samples)
    variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    variants = filter_by_missing_data(variants, max_allowed_missing_rate=0.99)
    assert list(variants.samples) == samples
    chunk = next(variants.iter_vars_chunks())
    assert numpy.all(gts[[True, True, True], :] == chunk.gts.gt_values)

    orig_vars = Variants.from_gt_array(gts, samples=samples)
    variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(variants.iter_vars_chunks())
    assert numpy.all(gts[[True, True, True], :] == chunk.gts.gt_values)


def _stack_gts(variants):
    return numpy.vstack([chunk.gts.gt_values for chunk in variants.iter_vars_chunks()])


GTS_FOR_REITERATION = numpy.array(
    [
        [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
        [[0, 1], [1, 0], [2, 2], [1, 0], [0, 0]],
        [[1, 1], [2, 2], [0, 1], [0, 0], [1, 2]],
        [[0, 0], [0, 1], [1, 1], [0, 2], [2, 0]],
    ]
)
SAMPLES_FOR_REITERATION = [0, 1, 2, 3, 4]


def _create_vars_for_reiteration(chunk_size):
    return Variants(
        _FromGtListChunkIterFactory(
            gts=[GTS_FOR_REITERATION], samples=SAMPLES_FOR_REITERATION
        ),
        desired_num_vars_per_chunk=chunk_size,
    )


def test_filtered_vars_can_be_iterated_more_than_once():
    gts = GTS_FOR_REITERATION
    for chunk_size in range(1, 5):
        variants = filter_by_missing_data(
            _create_vars_for_reiteration(chunk_size), max_allowed_missing_rate=1
        )
        assert numpy.all(_stack_gts(variants) == gts)
        assert numpy.all(_stack_gts(variants) == gts)

        # the stats are the ones of the last pass, they are not accumulated
        stats = gather_filtering_stats(variants)
        assert stats == {"missing_data": FilteringStats(vars_processed=4, vars_kept=4)}


def test_chained_filters_can_be_iterated_more_than_once():
    gts = GTS_FOR_REITERATION
    for chunk_size in range(1, 5):
        variants = filter_by_missing_data(
            _create_vars_for_reiteration(chunk_size), max_allowed_missing_rate=1
        )
        variants = filter_by_maf(variants, max_allowed_maf=0.99)
        assert numpy.all(_stack_gts(variants) == gts)
        assert numpy.all(_stack_gts(variants) == gts)

        stats = gather_filtering_stats(variants)
        assert stats == {
            "missing_data": FilteringStats(vars_processed=4, vars_kept=4),
            "maf": FilteringStats(vars_processed=4, vars_kept=4),
        }


def test_ld_filtered_vars_can_be_iterated_more_than_once():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 0], [2, 0], [1, 0], [0, 0]],
            [[1, 0], [0, 2], [0, 1], [0, 0], [2, 2]],
            [[1, 0], [0, 2], [0, 1], [0, 0], [2, 2]],
        ]
    )
    expected = gts[[True, False, False, True, False], :]
    for chunk_size in range(1, 6):
        orig_vars = Variants(
            _FromGtListChunkIterFactory(gts=[gts], samples=[0, 1, 2, 3, 4]),
            desired_num_vars_per_chunk=chunk_size,
        )
        variants = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
        # the second pass has to start with no reference genotype, as the
        # first one did
        assert numpy.all(_stack_gts(variants) == expected)
        assert numpy.all(_stack_gts(variants) == expected)

        stats = gather_filtering_stats(variants)
        assert stats == {"ld_and_maf": FilteringStats(vars_processed=5, vars_kept=2)}


def test_asking_for_the_metadata_does_not_consume_the_vars():
    gts = GTS_FOR_REITERATION
    for chunk_size in range(1, 5):
        variants = filter_by_missing_data(
            _create_vars_for_reiteration(chunk_size), max_allowed_missing_rate=1
        )
        assert variants.num_samples == 5
        assert variants.ploidy == 2
        assert list(variants.samples) == SAMPLES_FOR_REITERATION
        assert numpy.all(_stack_gts(variants) == gts)


def test_sample_filter_metadata_does_not_depend_on_the_call_order():
    kept_samples = [0, 1, 2]

    # metadata asked for before iterating
    variants = filter_samples(_create_vars_for_reiteration(2), samples=kept_samples)
    assert list(variants.samples) == kept_samples
    assert variants.num_samples == 3
    assert numpy.all(_stack_gts(variants) == GTS_FOR_REITERATION[:, :3, :])

    # metadata asked for after iterating
    variants = filter_samples(_create_vars_for_reiteration(2), samples=kept_samples)
    assert numpy.all(_stack_gts(variants) == GTS_FOR_REITERATION[:, :3, :])
    assert list(variants.samples) == kept_samples
    assert variants.num_samples == 3


def test_several_vars_can_share_one_filtered_source():
    # calc_ld_and_dist_per_pop creates one Variants per pop on top of a common
    # one, and every one of them has to see all the variations
    gts = GTS_FOR_REITERATION
    for chunk_size in range(1, 5):
        common_vars = filter_by_missing_data(
            _create_vars_for_reiteration(chunk_size), max_allowed_missing_rate=1
        )
        pop1_vars = filter_samples(common_vars, samples=[0, 1, 2])
        pop2_vars = filter_samples(common_vars, samples=[2, 3, 4])
        assert numpy.all(_stack_gts(pop1_vars) == gts[:, :3, :])
        assert numpy.all(_stack_gts(pop2_vars) == gts[:, 2:, :])


def test_stats_are_calculated_on_all_vars_when_asked_for_twice():
    for chunk_size in range(1, 5):
        variants = filter_by_missing_data(
            _create_vars_for_reiteration(chunk_size), max_allowed_missing_rate=1
        )
        res1 = calc_per_var_distribs(variants, stats="obs_het").obs_het
        res2 = calc_per_var_distribs(variants, stats="obs_het").obs_het
        assert res1.hist_counts.sum().iloc[0] == 4
        assert numpy.all(res1.hist_counts == res2.hist_counts)
        assert numpy.allclose(res1.mean, res2.mean)


def test_filtered_samples_are_a_tuple():
    variants = _create_vars_for_reiteration(2)
    assert variants.samples == tuple(SAMPLES_FOR_REITERATION)

    filtered = filter_by_missing_data(variants, max_allowed_missing_rate=1)
    assert filtered.samples == tuple(SAMPLES_FOR_REITERATION)

    filtered = filter_samples(variants, samples=[0, 1, 2])
    assert filtered.samples == (0, 1, 2)
    assert next(filtered.iter_vars_chunks()).gts.samples == (0, 1, 2)
