from typing import Sequence

import numpy
import pandas

from .config import MISSING_ALLELE, DEFAULT_NAME_POP_ALL_INDIS


def _get_pop_masks(gts, pops):
    if pops is None:
        pop = DEFAULT_NAME_POP_ALL_INDIS
        mask = numpy.ones(shape=(gts.num_indis), dtype=bool)
        yield pop, mask
    else:
        for pop, indis_in_pop in pops.items():
            mask = numpy.isin(gts.indi_names, indis_in_pop)
            yield pop, mask


class Genotypes:
    def __init__(
        self,
        gt_array: numpy.ndarray,
        indi_names: Sequence[str] | Sequence[int] | None = None,
    ):
        if gt_array.ndim != 3:
            raise ValueError(
                "Genotype array should have three dimesions: variant x indi x ploidy"
            )
        gt_array.flags.writeable = False
        self._gt_array = gt_array

        if indi_names is None:
            indi_names = numpy.array(range(self.num_indis))
        else:
            indi_names = numpy.array(indi_names)
        indi_names.flags.writeable = False
        self._indi_names = indi_names

    @property
    def num_vars(self):
        return self._gt_array.shape[0]

    @property
    def num_indis(self):
        return self._gt_array.shape[1]

    @property
    def ploidy(self):
        return self._gt_array.shape[2]

    @property
    def indi_names(self):
        return self._indi_names

    @property
    def gt_array(self):
        return self._gt_array

    def _get_all_alleles(self):
        return sorted(numpy.unique(self._gt_array))

    @property
    def alleles(self):
        return sorted(set(self._get_all_alleles()).difference([MISSING_ALLELE]))

    def _count_alleles_per_var(
        self,
        pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
        calc_freqs: bool = False,
        min_num_genotypes=DEFAULT_NAME_POP_ALL_INDIS,
    ):
        gts = self
        alleles = gts.alleles

        gt_array = gts._gt_array
        pop_masks = _get_pop_masks(gts, pops)
        ploidy = gts.ploidy

        result = {}
        for pop_id, pop_mask in pop_masks:
            gts_for_pop = gt_array[:, pop_mask, :]

            allele_counts = numpy.empty(
                shape=(gts_for_pop.shape[0], len(alleles)), dtype=numpy.int64
            )
            missing_counts = None
            for idx, allele in enumerate([MISSING_ALLELE] + alleles):
                allele_counts_per_row = numpy.sum(gts_for_pop == allele, axis=(1, 2))
                if idx == 0:
                    missing_counts = pandas.Series(allele_counts_per_row)
                else:
                    allele_counts[:, idx - 1] = allele_counts_per_row
            allele_counts = pandas.DataFrame(allele_counts, columns=alleles)

            result[pop_id] = {
                "allele_counts": allele_counts,
                "missing_gts_per_var": missing_counts,
                "num_indis": numpy.sum(pop_mask),
            }

            if calc_freqs:
                expected_num_allelic_gts_in_snp = (
                    gts_for_pop.shape[1] * gts_for_pop.shape[2]
                )
                num_allelic_gts_per_snp = (
                    expected_num_allelic_gts_in_snp - missing_counts.values
                )
                num_allelic_gts_per_snp = num_allelic_gts_per_snp.reshape(
                    (num_allelic_gts_per_snp.shape[0], 1)
                )
                allelic_freqs_per_snp = allele_counts / num_allelic_gts_per_snp
                num_gts_per_snp = (
                    num_allelic_gts_per_snp.reshape((num_allelic_gts_per_snp.size,))
                    / ploidy
                )
                not_enough_data = num_gts_per_snp < min_num_genotypes
                allelic_freqs_per_snp[not_enough_data] = numpy.nan

                result[pop_id]["allelic_freqs"] = allelic_freqs_per_snp

        return result

    def get_mat_012(self, transform_to_biallelic=False):
        res = self._count_alleles_per_var()
        allele_counts = res[DEFAULT_NAME_POP_ALL_INDIS]["allele_counts"].values

        num_genotyped_alleles_per_var = allele_counts.sum(axis=1)
        if numpy.any(num_genotyped_alleles_per_var == 0):
            raise ValueError("There are variants that only have missing data")

        max_num_alleles = allele_counts.shape[1]
        if max_num_alleles > 2 and not transform_to_biallelic:
            raise ValueError(
                f"In order to get the 012 matrix you should pass transform_to_biallelic=True, because you have {max_num_alleles} in at least one variation"
            )

        major_alleles = numpy.argmax(allele_counts, axis=1)
        gt_array = self.gt_array
        gts012 = numpy.sum(gt_array != major_alleles[:, None, None], axis=2)
        gts012[numpy.any(gt_array == MISSING_ALLELE, axis=2)] = MISSING_ALLELE
        return gts012

    def select_indis_by_bool_mask(self, mask: Sequence[bool]):
        gt_array = self._gt_array[:, mask, :]
        indi_names = self.indi_names[mask]
        Cls = self.__class__
        return Cls(gt_array=gt_array, indi_names=indi_names)

    def select_indis_by_name(self, indi_names: Sequence[str] | Sequence[int]):
        indis_not_found = set(indi_names).difference(self.indi_names)
        if indis_not_found:
            raise ValueError(
                "Some individuals selected were not found in the GTs: ",
                ",".join(map(str, indis_not_found)),
            )

        mask = numpy.isin(self.indi_names, indi_names)
        return self.select_indis_by_bool_mask(mask)
