import numpy
from pynei.variants import Variants
from pynei.gt_counts import (
    _calc_gt_is_het,
    _calc_obs_het_per_var,
    calc_obs_het_stats_per_var,
    calc_major_allele_stats_per_var,
    _count_alleles_per_var,
    _calc_maf_per_var,
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

    pop_name = pynei.config.DEF_POP_NAME
    res = calc_obs_het_stats_per_var(vars, hist_kwargs={"num_bins": 4})
    assert numpy.allclose(res["mean"].loc[pop_name], [0.5])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][pop_name] == [0, 1, 1, 0])


def test_count_alleles_per_var():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [0, 4], [0, 1], [1, 3]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    vars = Variants.from_gt_array(gts, samples=[1, 2, 3, 4])
    chunk = next(vars.iter_vars_chunks())

    res = _count_alleles_per_var(
        chunk, pops={0: [0, 1, 2, 3]}, calc_freqs=True, min_num_samples=1
    )
    assert res["alleles"] == {0, 1, 3, 4}
    assert numpy.all(res["counts"][0]["missing_gts_per_var"] == [2, 1, 8])
    expected_counts = [3, 3, 0, 0], [2, 3, 1, 1], [0, 0, 0, 0]
    assert numpy.all(res["counts"][0]["allele_counts"].values == expected_counts)
    expected_freqs = [
        [
            0.5,
            0.5,
            0.0,
            0.0,
        ],
        [0.28571429, 0.42857143, 0.14285714, 0.14285714],
        [numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    ]
    assert numpy.allclose(
        res["counts"][0]["allelic_freqs"].values, expected_freqs, equal_nan=True
    )

    res = _count_alleles_per_var(
        chunk, pops={0: [0, 1, 2, 3]}, calc_freqs=True, min_num_samples=3.1
    )
    assert res["alleles"] == {0, 1, 3, 4}
    assert numpy.all(res["counts"][0]["missing_gts_per_var"] == [2, 1, 8])
    expected_counts = [3, 3, 0, 0], [2, 3, 1, 1], [0, 0, 0, 0]
    assert numpy.all(res["counts"][0]["allele_counts"].values == expected_counts)
    expected_freqs = [
        [numpy.nan, numpy.nan, numpy.nan, numpy.nan],
        [0.28571429, 0.42857143, 0.14285714, 0.14285714],
        [numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    ]
    assert numpy.allclose(
        res["counts"][0]["allelic_freqs"], expected_freqs, equal_nan=True
    )

    pop_name = pynei.config.DEF_POP_NAME
    numpy.random.seed(42)
    gt_array = numpy.random.randint(0, 2, size=(2, 3, 2))
    vars = Variants.from_gt_array(gt_array)
    chunk = next(vars.iter_vars_chunks())
    res = _count_alleles_per_var(chunk, calc_freqs=False, min_num_samples=1)
    assert numpy.all(
        res["counts"][pop_name]["allele_counts"].values == [[4, 2], [5, 1]]
    )
    assert numpy.all(res["counts"][pop_name]["missing_gts_per_var"] == [0, 0])

    gt_array = numpy.random.randint(-1, 2, size=(2, 10, 2))
    vars = Variants.from_gt_array(gt_array)
    chunk = next(vars.iter_vars_chunks())
    res = _count_alleles_per_var(chunk, calc_freqs=True, min_num_samples=7)
    assert numpy.all(
        res["counts"][pop_name]["allele_counts"].values == [[7, 6], [7, 8]]
    )
    assert numpy.all(res["counts"][pop_name]["missing_gts_per_var"] == [7, 5])
    assert numpy.allclose(
        res["counts"][pop_name]["allelic_freqs"].values,
        [[numpy.nan, numpy.nan], [0.46666667, 0.53333333]],
        equal_nan=True,
    )


def test_maf_stats():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [1, 0], [0, 1], [1, 0]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    vars = Variants.from_gt_array(gts, samples=[1, 2, 3, 4])
    chunk = next(vars.iter_vars_chunks())
    res = _calc_maf_per_var(chunk, pops={0: slice(None, None)}, min_num_samples=1)
    assert numpy.allclose(
        res["major_allele_freqs_per_var"][0].values,
        [0.5, 0.57142857, numpy.nan],
        equal_nan=True,
    )

    res = calc_major_allele_stats_per_var(
        vars, hist_kwargs={"num_bins": 4}, min_num_samples=1
    )
    pop_name = pynei.config.DEF_POP_NAME
    assert numpy.allclose(res["mean"].loc[pop_name], [0.535714])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][pop_name] == [0, 0, 2, 0])
