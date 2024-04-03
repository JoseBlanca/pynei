import pytest
import numpy

from pynei import Genotypes, MISSING_ALLELE
from pynei import genotypes


def test_genotypes():
    num_vars = 3
    num_indis = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_indis, ploidy))
    gt_array[0, 0, 0] = 0
    gt_array[0, 0, 1] = 1
    gt_array[1, 0, 1] = 2
    gt_array[1, 0, 0] = MISSING_ALLELE
    gts = Genotypes(gt_array)
    assert gts.num_vars == num_vars
    assert gts.num_indis == num_indis
    assert gts.ploidy == ploidy
    assert gts.alleles == [0, 1, 2]


def test_filter_in_indis():
    gt_array = numpy.random.randint(0, 2, size=(2, 3, 2))
    indis = list(range(gt_array.shape[1]))
    gts = Genotypes(gt_array, indi_names=indis)
    selected_indis = indis[:2]
    gts = gts.select_indis_by_name(selected_indis)
    assert list(gts.indi_names) == selected_indis

    with pytest.raises(ValueError):
        gts.select_indis_by_name([3])


def test_count_alleles():
    numpy.random.seed(42)
    gt_array = numpy.random.randint(0, 2, size=(2, 3, 2))
    gts = Genotypes(gt_array)
    res = gts._count_alleles_per_var()
    assert numpy.all(
        res[genotypes.DEFAULT_NAME_POP_ALL_INDIS]["allele_counts"].values
        == [[4, 2], [5, 1]]
    )
    assert numpy.all(
        res[genotypes.DEFAULT_NAME_POP_ALL_INDIS]["missing_gts_per_var"].values
        == [0, 0]
    )

    gt_array = numpy.random.randint(-1, 2, size=(2, 10, 2))
    gts = Genotypes(gt_array)
    res = gts._count_alleles_per_var(calc_freqs=True, min_num_genotypes=7)
    pop_id = genotypes.DEFAULT_NAME_POP_ALL_INDIS
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
    res = gts.calc_major_allele_freqs(
        min_num_genotypes=3, pops={"pop1": list(range(5)), "pop2": list(range(5, 11))}
    )
    expected = [[0.57142857, 0.55555556], [0.55555556, numpy.nan], [0.4, 0.5]]
    assert numpy.allclose(res.values, numpy.array(expected), equal_nan=True)
    assert list(res.columns) == ["pop1", "pop2"]
