import numpy


class Genotypes:
    def __init__(
        self, gt_array: numpy.ndarray, indi_names: list[str | int] | None = None
    ):
        if gt_array.ndim != 3:
            raise ValueError(
                "Genotype array should have three dimesions: variant x indi x ploidy"
            )

        self._gt_array = gt_array

        if indi_names is None:
            indi_names = list(range(self.num_indis))
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
