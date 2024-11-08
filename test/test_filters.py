import pytest
import numpy

from pynei.variants import Variants
from pynei.var_filters import filter_by_missing_data, filter_by_maf


def test_filter_missing():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    orig_vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars = filter_by_missing_data(orig_vars, max_missing_rate=0)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, False, False], ...] == filtered_gts)

    vars = filter_by_missing_data(orig_vars, max_missing_rate=0.5)
    filtered_gts = numpy.ma.getdata(next(vars.iter_vars_chunks())._gt_array._gts)
    assert numpy.all(gts[[True, True, False], ...] == filtered_gts)

    vars = filter_by_missing_data(orig_vars, max_missing_rate=-0.1)
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
