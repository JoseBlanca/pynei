import numpy

from pynei import Genotypes


def test_genotypes():
    num_vars = 3
    num_indis = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_indis, ploidy))
    gts = Genotypes(gt_array)
    assert gts.num_vars == num_vars
    assert gts.num_indis == num_indis
    assert gts.ploidy == ploidy
