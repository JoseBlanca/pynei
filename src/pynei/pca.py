import numpy
import pandas

from .genotypes import Genotypes
from .dists import Distances


def _create_pc_names(num_prin_comps):
    n_digits = num_prin_comps // 10
    fstring = "{:0" + str(n_digits) + "d}"
    prin_comps_names = ["PC" + fstring.format(idx) for idx in range(num_prin_comps)]
    return prin_comps_names


def do_pca(data: pandas.DataFrame, center_data=True, standarize_data=True):
    if numpy.any(numpy.isnan(data)):
        raise ValueError("data can have no nan values")

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
    prin_comps_names = _create_pc_names(num_prin_comps)

    eig_vals = numpy.square(singular_vals) / (num_samples - 1)
    pcnts = eig_vals / eig_vals.sum() * 100.0
    projections = numpy.dot(prin_comps, data.T).T

    return {
        "projections": pandas.DataFrame(
            projections, index=sample_names, columns=prin_comps_names
        ),
        "explained_variance (%)": pandas.Series(pcnts, index=prin_comps_names),
        "princomps": pandas.DataFrame(
            prin_comps, index=prin_comps_names, columns=trait_names
        ),
    }


def do_pca_with_genotypes(gts: Genotypes, transform_to_biallelic=False):
    mat012 = gts.get_mat_012(transform_to_biallelic=transform_to_biallelic)
    mat012 = pandas.DataFrame(mat012.T, index=gts.indi_names)
    return do_pca(mat012, center_data=True, standarize_data=True)


def _make_f_matrix(matrix):
    """It takes an E matrix and returns an F matrix

    The input is the output of make_E_matrix

    For each element in matrix subtract mean of corresponding row and
    column and add the mean of all elements in the matrix
    """
    num_rows, num_cols = matrix.shape
    # make a vector of the means for each row and column
    # column_means = (numpy.add.reduce(E_matrix) / num_rows)
    column_means = (numpy.add.reduce(matrix) / num_rows)[:, numpy.newaxis]
    trans_matrix = numpy.transpose(matrix)
    row_sums = numpy.add.reduce(trans_matrix)
    row_means = row_sums / num_cols
    # calculate the mean of the whole matrix
    matrix_mean = numpy.sum(row_sums) / (num_rows * num_cols)
    # adjust each element in the E matrix to make the F matrix

    matrix -= row_means
    matrix -= column_means
    matrix += matrix_mean

    return matrix


def do_pcoa(dists: Distances):
    "It does a Principal Coordinate Analysis on a distance matrix"
    # the code for this function is taken from pycogent metric_scaling.py
    # Principles of Multivariate analysis: A User's Perspective.
    # W.J. Krzanowski Oxford University Press, 2000. p106.

    sample_names = dists.names
    dists = dists.square_dists.values

    if numpy.any(numpy.isnan(dists)):
        raise ValueError("dists array has nan values")

    e_matrix = (dists * dists) / -2.0
    f_matrix = _make_f_matrix(e_matrix)

    eigvals, eigvecs = numpy.linalg.eigh(f_matrix)
    eigvecs = eigvecs.transpose()
    # drop imaginary component, if we got one
    eigvals, eigvecs = eigvals.real, eigvecs.real

    # convert eigvals and eigvecs to point matrix
    # normalized eigenvectors with eigenvalues

    # get the coordinates of the n points on the jth axis of the Euclidean
    # representation as the elements of (sqrt(eigvalj))eigvecj
    # must take the absolute value of the eigvals since they can be negative
    pca_matrix = eigvecs * numpy.sqrt(abs(eigvals))[:, numpy.newaxis]

    # output
    # get order to output eigenvectors values. reports the eigvecs according
    # to their cooresponding eigvals from greatest to least
    vector_order = list(numpy.argsort(eigvals))
    vector_order.reverse()

    eigvals = eigvals[vector_order]

    # eigenvalues
    pcnts = (eigvals / numpy.sum(eigvals)) * 100.0

    # the outputs
    # eigenvectors in the original pycogent implementation, here we name them
    # princoords
    # I think that we're doing: if the eigenvectors are written as columns,
    # the rows of the resulting table are the coordinates of the objects in
    # PCO space
    projections = []
    for name_i in range(dists.shape[0]):
        eigvect = [pca_matrix[vec_i, name_i] for vec_i in vector_order]
        projections.append(eigvect)
    projections = numpy.array(projections)
    prin_comps_names = _create_pc_names(projections.shape[1])

    return {
        "projections": pandas.DataFrame(
            projections, index=sample_names, columns=prin_comps_names
        ),
        "explained_variance (%)": pandas.Series(pcnts, index=prin_comps_names),
    }
