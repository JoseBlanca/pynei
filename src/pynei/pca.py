from functools import partial

import numpy
import pandas

from pynei.gt_counts import _count_alleles_per_var
from pynei.config import DEF_POP_NAME, MISSING_ALLELE
from pynei.pipeline import Pipeline


def _create_012_gt_matrix(chunk, transform_to_biallelic=False):
    res = _count_alleles_per_var(chunk, calc_freqs=False)
    allele_counts = res["counts"][DEF_POP_NAME]["allele_counts"].values

    num_genotyped_alleles_per_var = allele_counts.sum(axis=1)
    if numpy.any(num_genotyped_alleles_per_var == 0):
        raise ValueError("There are variants that only have missing data")

    max_num_alleles = allele_counts.shape[1]
    if max_num_alleles > 2 and not transform_to_biallelic:
        raise ValueError(
            f"In order to get the 012 matrix you should pass transform_to_biallelic=True, because you have {max_num_alleles} in at least one variation"
        )

    major_alleles = numpy.argmax(allele_counts, axis=1)
    gt_array = chunk.gts.gt_array
    gts012 = numpy.sum(gt_array != major_alleles[:, None, None], axis=2)
    gts012[numpy.any(gt_array == MISSING_ALLELE, axis=2)] = MISSING_ALLELE
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


def do_pca_with_vars(variants, transform_to_biallelic=False):
    mat012 = create_012_gt_matrix(
        variants, transform_to_biallelic=transform_to_biallelic
    )
    mat012 = pandas.DataFrame(mat012.T, index=variants.samples)
    return do_pca(mat012, center_data=True, standarize_data=True)
