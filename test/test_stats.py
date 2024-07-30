import numpy

from pynei import (
    Genotypes,
    calc_major_allele_freqs,
    calc_obs_het,
    calc_poly_vars_ratio,
    calc_exp_het,
    calc_allele_freq_spectrum,
)
from pynei.config import DEFAULT_NAME_POP_ALL_INDIS


def test_calc_major_allele_freqs():
    numpy.random.seed(42)
    gt_array = numpy.random.randint(-1, 3, size=(3, 10, 2))
    gts = Genotypes(gt_array)

    res = calc_poly_vars_ratio(
        gts,
        poly_threshold=0.51,
        min_num_genotypes=3,
        pops={"pop1": list(range(5)), "pop2": list(range(5, 11))},
    )
    assert numpy.allclose(res["poly_ratio"].values, [0.333333, 0.5])
    assert numpy.allclose(res["num_variable"].values, [3, 2])
