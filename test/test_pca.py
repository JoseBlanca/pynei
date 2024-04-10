import numpy

from datasets import IRIS

from pynei import do_pca, Genotypes, do_pca_with_genotypes


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


def test_pca_gt():
    numpy.random.seed(seed=42)
    num_vars = 100
    num_indis = 20
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_indis, ploidy))
    gts = Genotypes(gt_array)
    do_pca_with_genotypes(gts)
