import numpy
from pynei.variants import Variants
from pynei.gt_counts import (
    _calc_gt_is_het,
    _calc_obs_het_per_var,
    calc_obs_het_stats_per_var,
)
import pynei.config


def test_obs_het_stats():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [0, 0], [0, 1], [1, 0]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    vars = Variants.from_gt_array(gts, samples=[1, 2, 3, 4])
    chunk = next(vars.iter_vars_chunks())

    res = _calc_gt_is_het(chunk)
    expected = [
        [False, False, True, False],
        [True, False, False, False],
        [True, True, True, True],
    ]
    assert numpy.array_equal(res["gt_is_missing"], expected)
    expected = [
        [False, False, False, True],
        [False, False, True, True],
        [False, False, False, False],
    ]
    assert numpy.array_equal(res["gt_is_het"], expected)

    # pandas pops are columns
    res = _calc_obs_het_per_var(chunk, pops={"pop": slice(None, None)})
    numpy.allclose(res["obs_het_per_var"].values, [0.33333333, 0.66666667, numpy.nan])
    res = _calc_obs_het_per_var(
        chunk,
        pops={
            "pop1": [0, 1],
            "pop2": [2, 3],
        },
    )
    assert sorted(res["obs_het_per_var"].columns) == ["pop1", "pop2"]
    assert res["obs_het_per_var"].values.shape == (3, 2)

    res = calc_obs_het_stats_per_var(vars, hist_kwargs={"num_bins": 4})
    pop_name = pynei.config.DEF_POP_NAME
    assert numpy.allclose(res["mean"].loc[pop_name], [0.5])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][pop_name] == [0, 1, 1, 0])
