import numpy
import pandas

from .genotypes import Genotypes


def do_pca(data: pandas.DataFrame, center_data=True, standarize_data=True):
    if standarize_data and not center_data:
        raise ValueError("If you standarize you have to also center the data")

    trait_names = data.columns
    sample_names = data.index

    data = data.values
    num_samples, num_traits = data.shape

    if center_data:
        data = data - data.mean(axis=0)

    if standarize_data:
        data = data / data.std(axis=0)

    U, Sigma, Vh = numpy.linalg.svd(data, full_matrices=False)
    singular_vals = Sigma
    prin_comps = Vh
    num_prin_comps = prin_comps.shape[0]
    n_digits = num_prin_comps // 10
    fstring = "{:0" + str(n_digits) + "d}"
    prin_comps_names = ["PC" + fstring.format(idx) for idx in range(num_prin_comps)]

    eig_vals = numpy.square(singular_vals) / (num_samples - 1)
    pcnts = eig_vals / eig_vals.sum() * 100.0
    projections = numpy.dot(prin_comps, data.T).T

    return {
        "projections": pandas.DataFrame(
            projections, index=sample_names, columns=prin_comps_names
        ),
        "explained_variance": pandas.Series(pcnts, index=prin_comps_names),
        "princomps": pandas.DataFrame(
            prin_comps, index=prin_comps_names, columns=trait_names
        ),
    }


def do_pca_with_genotypes(gts: Genotypes, transform_to_biallelic=False):
    mat012 = gts.get_mat_012(transform_to_biallelic=transform_to_biallelic)
    mat012 = pandas.DataFrame(mat012.T, index=gts.indi_names)
    return do_pca(mat012, center_data=True, standarize_data=True)
