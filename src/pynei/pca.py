from functools import partial
import numpy

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
