import numpy
import pandas

from .config import (
    PANDAS_STRING_STORAGE,
    PANDAS_FLOAT_DTYPE,
    PANDAS_INT_DTYPE,
)


def is_read_only(arr: numpy.array) -> bool:
    return not arr.flags.writeable


def _normalize_pandas_types(dframe):
    new_dframe = {}
    for col, values in dframe.items():
        if pandas.api.types.is_string_dtype(values):
            values = pandas.Series(
                values, dtype=pandas.StringDtype(PANDAS_STRING_STORAGE)
            )
        elif pandas.api.types.is_float_dtype(values):
            values = pandas.Series(values, dtype=PANDAS_FLOAT_DTYPE())
        elif pandas.api.types.is_integer_dtype(values):
            values = pandas.Series(values, dtype=PANDAS_INT_DTYPE())
        else:
            raise ValueError(f"Unsupported dtype for column {col}")
        values.flags.writeable = False
        new_dframe[col] = values
    return pandas.DataFrame(new_dframe)


class VariantsChunk:
    def __init__(
        self,
        gts: numpy.array,
        variants_info: pandas.DataFrame | None = None,
        alleles: pandas.DataFrame | None = None,
        samples: list[str] | None = None,
    ):
        if not isinstance(gts, numpy.ndarray):
            raise ValueError("gts must be a numpy array")
        if not numpy.issubdtype(gts.dtype, numpy.integer):
            raise ValueError("gts must be an integer numpy array")
        if gts.ndim != 3:
            raise ValueError(
                "gts must be a 3D numpy array: (num_vars, num_samples, ploidy)"
            )

        if not is_read_only(gts):
            gts = gts.copy()
            gts.flags.writeable = False

        if variants_info is not None:
            if variants_info.shape[0] != gts.shape[0]:
                raise ValueError(
                    "variants_info must have the same number of rows as gts"
                )
            variants_info = _normalize_pandas_types(variants_info)

        if alleles is not None:
            if alleles.shape[0] != gts.shape[0]:
                raise ValueError("alleles must have the same number of rows as gts")

        if samples is not None:
            if len(samples) != gts.shape[1]:
                raise ValueError("There has to be as many samples gts.shape[1]")

        self.gt_array = gts
        self.variants_info = variants_info
        self.alleles = alleles
        self.samples = samples

    @property
    def num_vars(self):
        return self.gt_array.shape[0]

    @property
    def num_samples(self):
        return self.gt_array.shape[1]

    @property
    def ploidy(self):
        return self.gt_array.shape[2]

    @property
    def gts(self):
        return self.gts
