import numpy

from pynei.variants import Variants
from pynei.sample_stats import calc_per_sample_stats


def test_filter_missing():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    vars.desired_num_vars_per_chunk = 2
    res = calc_per_sample_stats(vars)
    expected = [0.33333, 0.33333, 0.33333, 0.33333, 0.666667]
    assert numpy.allclose(res["missing_gt_rate"], expected)

    expected = [0.0, 0.33333, 0.33333, 0.33333, 0.0]
    assert numpy.allclose(res["obs_het_rate"], expected)
