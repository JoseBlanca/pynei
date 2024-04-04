import pytest
import numpy

from pynei import Genotypes
from pynei.config import MISSING_ALLELE


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
