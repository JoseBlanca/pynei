import pytest
import numpy

from pynei.variants import Variants
from pynei.var_filters import (
    filter_by_missing_data,
    filter_by_maf,
    filter_by_obs_het,
    gather_filtering_stats,
    filter_samples,
    filter_by_ld_and_maf,
)


def test_filter_missing():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_missing_data(orig_vars, max_allowed_missing_rate=0)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, False], ...] == filtered_gts)

    vars = filter_by_missing_data(orig_vars, max_allowed_missing_rate=0.5)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, True, False], ...] == filtered_gts)

    vars = filter_by_missing_data(orig_vars, max_allowed_missing_rate=-0.1)
    with pytest.raises(StopIteration):
        next(vars.iter_vars_chunks())


def test_filter_mafs():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_maf(orig_vars, max_allowed_maf=0.9)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, True, False], ...] == filtered_gts)

    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_maf(orig_vars, max_allowed_maf=0.7)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[False, True, False], ...] == filtered_gts)

    stats = gather_filtering_stats(vars)
    assert stats == {"maf": {"vars_processed": 3, "vars_kept": 1}}


def test_filter_obs_het():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_obs_het(orig_vars, max_allowed_obs_het=1.5 / 5.0)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, True], ...] == filtered_gts)

    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_obs_het(orig_vars, max_allowed_obs_het=0)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[False, False, True], ...] == filtered_gts)

    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_obs_het(orig_vars, max_allowed_obs_het=4 / 5)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
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
    vars = filter_by_obs_het(orig_vars, max_allowed_obs_het=1.5 / 5.0)
    assert vars.num_samples == 5
    assert vars.ploidy == 2
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, True], ...] == filtered_gts)

    # after filtering
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_obs_het(orig_vars, max_allowed_obs_het=1.5 / 5.0)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, True], ...] == filtered_gts)
    assert vars.num_samples == 5
    assert vars.ploidy == 2


def test_filter_samples():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_samples(orig_vars, samples=[0, 1, 2])
    assert numpy.all(next(vars.iter_vars_chunks()).gts.gt_values == gts[:, :3, :])

    vars = filter_samples(orig_vars, samples=slice(3))
    assert numpy.all(next(vars.iter_vars_chunks()).gts.gt_values == gts[:, :3, :])


def test_filter_ld():
    gts = numpy.array(
        [
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    with pytest.raises(StopIteration):
        next(vars.iter_vars_chunks())

    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 0], [2, 0], [1, 0], [0, 0]],
            [[1, 0], [0, 2], [0, 1], [0, 0], [2, 2]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(vars.iter_vars_chunks())
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
    vars = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(vars.iter_vars_chunks())
    assert numpy.all(gts[[True, False, True, True, False], :] == chunk.gts.gt_values)

    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(vars.iter_vars_chunks())
    assert numpy.all(gts[[True, True, False], :] == chunk.gts.gt_values)

    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_ld_and_maf(orig_vars, max_allowed_maf=0.9)
    chunk = next(vars.iter_vars_chunks())
    assert numpy.all(gts[[True, False, False], :] == chunk.gts.gt_values)

    return
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, True, False], ...] == filtered_gts)

    return
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[False, True, False], ...] == filtered_gts)

    stats = gather_filtering_stats(vars)
    assert stats == {"maf": {"vars_processed": 3, "vars_kept": 1}}
