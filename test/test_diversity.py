import numpy

from pynei.variants import Variants
from pynei.diversity import (
    _calc_exp_het_per_var,
    _calc_unbiased_exp_het_per_var,
    calc_exp_het_stats_per_var,
    _calc_num_poly_vars,
    calc_poly_vars_ratio_per_var,
)
from pynei.config import DEF_POP_NAME


def test_calc_exp_het():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, -1]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    #       allele counts  allele freqs            exp hom                          exp het      u exp het
    #       pop1    pop2   pop1         pop2       pop1               pop2          pop1   pop2  pop1   pop2
    # snp1  2 1 1   5 0 0  2/4 1/4 1/4  5/5 0   0  0.25 0.0625 0.0625 1    0    0   0.625  0     0.8333 0
    # snp2  4 0 0   2 2 0  4/4 0   0    2/4 2/4 0  1    0      0      0.25 0.25 0   0      0.5   0      0.6666
    # snp3  0 0 0   0 0 0  nan nan nan  nan nan nan nan nan    nan    nan  nan  nan nan    nan   nan    nan
    vars = Variants.from_gt_array(gts, samples=[0, 1, 2, 3, 4])
    chunk = next(vars.iter_vars_chunks())

    pops = {1: [0, 1], 2: [2, 3, 4]}
    res = _calc_exp_het_per_var(chunk, pops, min_num_samples=1)
    expected = [[0.625, 0.0], [0.0, 0.5], [numpy.nan, numpy.nan]]
    assert numpy.allclose(res["exp_het"].values, expected, equal_nan=True)
    expected = [[0, 1], [0, 2], [4, 6]]
    assert numpy.all(res["missing_allelic_gts"].values == expected)

    res = _calc_unbiased_exp_het_per_var(chunk, pops, min_num_samples=1)
    expected = [[0.83333333, 0.0], [0.0, 0.66666667], [numpy.nan, numpy.nan]]
    assert numpy.allclose(res["exp_het"].values, expected, equal_nan=True)

    res = calc_exp_het_stats_per_var(
        vars, hist_kwargs={"num_bins": 4}, min_num_samples=1
    )
    assert numpy.allclose(res["mean"].loc[DEF_POP_NAME], [0.422619])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][DEF_POP_NAME] == [0, 2, 0, 0])


def test_poly_vars_ratio():
    numpy.random.seed(42)
    gt_array = numpy.random.randint(-1, 3, size=(3, 10, 2))
    vars = Variants.from_gt_array(gt_array, samples=list(range(0, 10)))
    chunk = next(vars.iter_vars_chunks())

    res = _calc_num_poly_vars(
        chunk,
        poly_threshold=0.51,
        min_num_samples=3,
        pops={"pop1": list(range(5)), "pop2": list(range(5, 10))},
    )
    assert numpy.allclose(res["num_variable"].values, [3, 2])
    assert numpy.allclose(res["num_poly"].values, [1, 1])

    vars.desired_num_vars_per_chunk = 2
    res = calc_poly_vars_ratio_per_var(
        vars,
        poly_threshold=0.51,
        min_num_samples=3,
        pops={"pop1": list(range(5)), "pop2": list(range(5, 10))},
    )
    assert numpy.allclose(res["poly_ratio_over_variables"].values, [0.333333, 0.5])
    assert numpy.allclose(res["poly_ratio"].values, [0.333333, 0.5])
