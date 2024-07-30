import pytest
import numpy

from pynei.pca import _create_012_gt_matrix, create_012_gt_matrix
from pynei.variants import Variants
from pynei.config import MISSING_ALLELE


def test_mat012():
    num_vars = 3
    num_indis = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 1, size=(num_vars, num_indis, ploidy))
    gt_array[0, 0, 0] = 0
    gt_array[0, 0, 1] = 1
    gt_array[1, :, :] = 2
    gt_array[1, 0, 1] = 3
    gt_array[2, :, :] = MISSING_ALLELE

    vars = Variants.from_gt_array(gt_array)
    chunk = next(vars.iter_vars_chunks())
    with pytest.raises(ValueError):
        _create_012_gt_matrix(chunk)

    gt_array = numpy.random.randint(0, 1, size=(num_vars, num_indis, ploidy))
    gt_array[0, 0, 0] = 0
    gt_array[0, 0, 1] = 1
    gt_array[0, 1, 0] = 1
    gt_array[0, 1, 1] = 1
    gt_array[1, :, :] = 2
    gt_array[1, 0, 1] = 3
    gt_array[1, 3, 0] = 3
    gt_array[1, 3, 1] = 3
    vars = Variants.from_gt_array(gt_array)
    chunk = next(vars.iter_vars_chunks())
    with pytest.raises(ValueError):
        _create_012_gt_matrix(chunk)

    mat012 = _create_012_gt_matrix(chunk, transform_to_biallelic=True)
    expected = [[1, 2, 0, 0], [1, 0, 0, 2], [0, 0, 0, 0]]
    assert numpy.all(mat012 == expected)

    vars.desired_num_vars_per_chunk = 2
    mat012_2 = create_012_gt_matrix(vars, transform_to_biallelic=True)
    assert numpy.array_equal(mat012, mat012_2)
