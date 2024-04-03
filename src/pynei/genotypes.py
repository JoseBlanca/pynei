from typing import Sequence

import numpy

MISSING_ALLELE = -1


class Genotypes:
    def __init__(
        self, gt_array: numpy.ndarray, indi_names: Sequence[str | int] | None = None
    ):
        if gt_array.ndim != 3:
            raise ValueError(
                "Genotype array should have three dimesions: variant x indi x ploidy"
            )

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
    def alleles(self):
        return sorted(set(numpy.unique(self._gt_array)).difference([MISSING_ALLELE]))

    def select_indis_by_bool_mask(self, mask: Sequence[bool]):
        gt_array = self._gt_array[:, mask, :]
        indi_names = self.indi_names[mask]
        Cls = self.__class__
        return Cls(gt_array=gt_array, indi_names=indi_names)

    def select_indis_by_name(self, indi_names: Sequence[str | int]):
        indis_not_found = set(indi_names).difference(self.indi_names)
        if indis_not_found:
            raise ValueError(
                "Some individuals selected were not found in the GTs: ",
                ",".join(map(str, indis_not_found)),
            )

        mask = numpy.isin(self.indi_names, indi_names)
        return self.select_indis_by_bool_mask(mask)
