from typing import Iterator, Self
import itertools

import numpy
import pandas

from .config import (
    PANDAS_STRING_STORAGE,
    PANDAS_FLOAT_DTYPE,
    PANDAS_INT_DTYPE,
    DEF_NUM_VARS_PER_CHUNK,
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

        self._gt_array = gts
        self._vars_info = variants_info
        self._alleles = alleles
        self.samples = samples

    @property
    def num_vars(self):
        return self._gt_array.shape[0]

    @property
    def num_samples(self):
        return self._gt_array.shape[1]

    @property
    def ploidy(self):
        return self._gt_array.shape[2]

    @property
    def gts(self):
        return self._gt_array


class Variants:
    def __init__(
        self, vars_chunks: Iterator[VariantsChunk], store_chunks_in_memory=False
    ):
        vars_chunks = iter(vars_chunks)
        if store_chunks_in_memory:
            self._vars_chunks = list(vars_chunks)
            self._chunks_iter = None
        else:
            self._vars_chunks = None
            self._chunks_iter = vars_chunks

    def _get_orig_vars_iter(self):
        if self._vars_chunks is not None:
            return iter(self._vars_chunks)
        else:
            return self._chunks_iter

    def _get_first_chunk(self):
        if self._vars_chunks is not None:
            chunk = self._vars_chunks[0]
        else:
            try:
                chunk = next(self._chunks_iter)
            except StopIteration:
                raise RuntimeError("No variants_chunks available")
            self._chunks_iter = itertools.chain([chunk], self._chunks_iter)
        return chunk

    def iter_vars_chunks(
        self, num_vars_per_chunk=DEF_NUM_VARS_PER_CHUNK
    ) -> Iterator[VariantsChunk]:
        return self._get_orig_vars_iter()

    @property
    def samples(self):
        return self._get_first_chunk().samples

    @classmethod
    def from_gt_array(
        cls,
        gts: numpy.array,
        samples: list[str] | None = None,
    ) -> Self:
        chunk = VariantsChunk(gts=gts, samples=samples)
        return cls(vars_chunks=[chunk], store_chunks_in_memory=True)
