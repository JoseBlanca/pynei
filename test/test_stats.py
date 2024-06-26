import numpy

from pynei import (
    Genotypes,
    calc_major_allele_freqs,
    calc_obs_het,
    calc_poly_vars_ratio,
    calc_exp_het,
)
from pynei.config import DEFAULT_NAME_POP_ALL_INDIS


def test_count_alleles():
    numpy.random.seed(42)
    gt_array = numpy.random.randint(0, 2, size=(2, 3, 2))
    gts = Genotypes(gt_array)
    res = gts._count_alleles_per_var()
    assert numpy.all(
        res[DEFAULT_NAME_POP_ALL_INDIS]["allele_counts"].values == [[4, 2], [5, 1]]
    )
    assert numpy.all(
        res[DEFAULT_NAME_POP_ALL_INDIS]["missing_gts_per_var"].values == [0, 0]
    )

    gt_array = numpy.random.randint(-1, 2, size=(2, 10, 2))
    gts = Genotypes(gt_array)
    res = gts._count_alleles_per_var(calc_freqs=True, min_num_genotypes=7)
    pop_id = DEFAULT_NAME_POP_ALL_INDIS
    assert numpy.all(res[pop_id]["allele_counts"].values == [[7, 6], [7, 8]])
    assert numpy.all(res[pop_id]["missing_gts_per_var"].values == [7, 5])
    assert numpy.allclose(
        res[pop_id]["allelic_freqs"].values,
        [[numpy.nan, numpy.nan], [0.46666667, 0.53333333]],
        equal_nan=True,
    )


def test_calc_major_allele_freqs():
    numpy.random.seed(42)
    gt_array = numpy.random.randint(-1, 3, size=(3, 10, 2))
    gts = Genotypes(gt_array)
    res = calc_major_allele_freqs(
        gts,
        min_num_genotypes=3,
        pops={"pop1": list(range(5)), "pop2": list(range(5, 11))},
    )
    expected = [[0.57142857, 0.55555556], [0.55555556, numpy.nan], [0.4, 0.5]]
    assert numpy.allclose(res.values, numpy.array(expected), equal_nan=True)
    assert list(res.columns) == ["pop1", "pop2"]

    poly_ratio = calc_poly_vars_ratio(
        gts,
        poly_threshold=0.51,
        min_num_genotypes=3,
        pops={"pop1": list(range(5)), "pop2": list(range(5, 11))},
    )
    assert numpy.allclose(poly_ratio.values, [0.333333, 0.5])


def test_calc_obs_het():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, -1]],
            [[0, 0], [0, 0], [0, 1], [1, -1], [-1, 0]],
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    gts = Genotypes(gts)
    expected_obs_het = numpy.nanmean(numpy.array([1 / 4, numpy.nan, numpy.nan]))
    obs_het = calc_obs_het(gts, min_num_genotypes=4)
    assert numpy.allclose(obs_het.values, expected_obs_het, equal_nan=True)
    assert list(obs_het.index) == [DEFAULT_NAME_POP_ALL_INDIS]


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
    gts = Genotypes(gts)
    pops = {"pop1": [0, 1], "pop2": [2, 3, 4]}
    nei = calc_exp_het(gts, pops=pops, min_num_genotypes=1)
    assert numpy.allclose(nei.values, [0.416667, 0.333333])

    nei = calc_exp_het(gts, pops=pops, min_num_genotypes=1, unbiased=False)
    assert numpy.allclose(nei.values, [0.3125, 0.2500])
