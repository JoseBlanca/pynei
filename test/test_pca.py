import pytest
import numpy

from pynei.pca import (
    _create_012_gt_matrix,
    create_012_gt_matrix,
    do_pca,
    do_pca_with_vars,
    do_pcoa,
    do_pcoa_with_vars,
)
from pynei.variants import Variants
from .datasets import IRIS
from pynei.dists import Distances
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
    missing_mask = numpy.zeros_like(gt_array, dtype=bool)
    missing_mask[2, :, :] = True
    gt_array = numpy.ma.array(gt_array, mask=missing_mask)

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


def test_pca():
    pca = do_pca(IRIS["characterization"])

    expected_princomps = [
        [0.52106591, -0.26934744, 0.5804131, 0.56485654],
        [-0.37741762, -0.92329566, -0.02449161, -0.06694199],
        [0.71956635, -0.24438178, -0.14212637, -0.63427274],
        [0.26128628, -0.12350962, -0.80144925, 0.52359713],
    ]
    princomps = pca["princomps"].values
    for idx in range(princomps.shape[0]):
        assert numpy.allclose(
            expected_princomps[idx], princomps[idx]
        ) or numpy.allclose(expected_princomps[idx], -princomps[idx])


def test_pca_vars():
    numpy.random.seed(seed=42)
    num_vars = 100
    num_indis = 20
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_indis, ploidy))
    vars = Variants.from_gt_array(gt_array)
    do_pca_with_vars(vars)


def test_pcoa():
    dists = [0.2, 0.3, 0.9, 0.9, 0.1, 0.8, 0.7, 0.7, 0.8, 0.2]
    dists = Distances(numpy.array(dists), names=["i1", "i2", "i3", "i4", "i5"])
    res = do_pcoa(dists)
    projections = res["projections"]
    assert abs(projections.loc["i1", "PC0"] - projections.loc["i2", "PC0"]) < abs(
        projections.loc["i1", "PC0"] - projections.loc["i4", "PC0"]
    )


def test_pcoa_with_vars():
    numpy.random.seed(seed=42)
    num_vars = 100
    num_indis = 20
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_indis, ploidy))
    vars = Variants.from_gt_array(gt_array)

    do_pcoa_with_vars(vars, use_approx_embedding_algorithm=True)
    do_pcoa_with_vars(vars, use_approx_embedding_algorithm=False)
