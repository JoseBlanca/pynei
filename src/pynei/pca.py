from dataclasses import dataclass
from functools import partial

import numpy
import pandas

from pynei.gt_counts import _count_alleles_per_var
from pynei.config import MISSING_ALLELE
from pynei.pipeline import Pipeline
from pynei.dists import Distances, calc_pairwise_kosman_dists


@dataclass(frozen=True)
class PCAResult:
    """The result of a Principal Component Analysis."""

    projections: pandas.DataFrame
    "The coordinates of every sample in the PCA space, one column per PC"

    explained_variance_percent: pandas.Series
    "The percentage of the variance that every PC explains"

    princomps: pandas.DataFrame
    "The principal components, one row per PC and one column per trait"


@dataclass(frozen=True)
class PCoAResult:
    """The result of a Principal Coordinate Analysis."""

    projections: pandas.DataFrame
    "The coordinates of every sample in the PCoA space, one column per PC"

    explained_variance_percent: pandas.Series
    "The percentage of the variance that every PC explains"


def _create_012_gt_matrix(chunk, transform_to_biallelic=False):
    gts012 = chunk.gts.to_012()

    if numpy.any(numpy.all(gts012 == MISSING_ALLELE, axis=1)):
        raise ValueError("There are variants that only have missing data")

    if not transform_to_biallelic:
        num_alleles = len(_count_alleles_per_var(chunk, calc_freqs=False)["alleles"])
        if num_alleles > 2:
            raise ValueError(
                f"In order to get the 012 matrix you should pass transform_to_biallelic=True, because you have {num_alleles} in at least one variation"
            )

    return gts012


def _append_array(array: numpy.ndarray | None, array_to_append: numpy.ndarray):
    if array is None:
        return array_to_append

    return numpy.vstack((array, array_to_append))


def create_012_gt_matrix(variants, transform_to_biallelic=False):
    create_012_matrix = partial(
        _create_012_gt_matrix, transform_to_biallelic=transform_to_biallelic
    )
    pipeline = Pipeline(map_functs=[create_012_matrix], reduce_funct=_append_array)
    return pipeline.map_and_reduce(variants)


def _create_pc_names(num_prin_comps):
    fstring = "{:0" + str(len(str(num_prin_comps))) + "d}"
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

    return PCAResult(
        projections=pandas.DataFrame(
            projections, index=sample_names, columns=prin_comps_names
        ),
        explained_variance_percent=pandas.Series(pcnts, index=prin_comps_names),
        princomps=pandas.DataFrame(
            prin_comps, index=prin_comps_names, columns=trait_names
        ),
    )


def _fill_missing_gts_with_var_mean(mat012):
    # Once the data is centered the mean of the variant is 0, so a missing
    # genotype does not pull the sample in any direction
    is_missing = mat012 == MISSING_ALLELE
    mat012 = numpy.where(is_missing, numpy.nan, mat012.astype(float))
    var_means = numpy.nanmean(mat012, axis=1)
    return numpy.where(is_missing, var_means[:, numpy.newaxis], mat012)


def do_pca_from_variants(variants, transform_to_biallelic=False):
    """It does a PCA using the 012 matrix of the variants.

    The missing genotypes are replaced by the mean of their variant.
    """
    mat012 = create_012_gt_matrix(
        variants, transform_to_biallelic=transform_to_biallelic
    )
    mat012 = _fill_missing_gts_with_var_mean(mat012)
    mat012 = pandas.DataFrame(mat012.T, index=variants.samples)
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

    return PCoAResult(
        projections=pandas.DataFrame(
            projections, index=sample_names, columns=prin_comps_names
        ),
        explained_variance_percent=pandas.Series(pcnts, index=prin_comps_names),
    )


def do_pcoa_from_variants(
    variants, min_num_snps=None, use_approx_embedding_algorithm=False
):
    dists = calc_pairwise_kosman_dists(
        variants,
        min_num_snps=min_num_snps,
        use_approx_embedding_algorithm=use_approx_embedding_algorithm,
    )
    return do_pcoa(dists)
