from typing import Iterator, Self, Sequence
from collections.abc import Sequence as SequenceABC
import itertools


import numpy
import pandas

from .config import (
    PANDAS_STRING_STORAGE,
    PANDAS_FLOAT_DTYPE,
    PANDAS_INT_DTYPE,
    DEF_NUM_VARS_PER_CHUNK,
)


class Genotypes:
    def __init__(
        self,
        gt_array: numpy.ndarray,
        samples: numpy.ndarray | Sequence[str] | None = None,
    ):
        if not isinstance(gt_array, numpy.ndarray):
            raise ValueError("gts must be a numpy array")
        if not numpy.issubdtype(gt_array.dtype, numpy.integer):
            raise ValueError("gts must be an integer numpy array")

        assert gt_array.ndim == 3

        if gt_array.flags.writeable:
            gt_array = gt_array.copy()
            gt_array.flags.writeable = False

        if samples is not None:
            samples = numpy.array(samples)
            samples.flags.writeable = False
            if len(set(samples)) < samples.size:
                unique_elements, counts = numpy.unique(samples, return_counts=True)
                duplicated_samples = unique_elements[counts > 1]
                raise ValueError(f"Duplicated sample names: {duplicated_samples}")

        else:
            samples = numpy.arange(gt_array.shape[1])
        samples.flags.writeable = False

        if gt_array.shape[1] != samples.size:
            raise ValueError(
                f"Number of samples in gts ({gt_array.shape[1]}) and number of given samples ({samples.size}) do not match"
            )

        self._gts = gt_array
        self._samples = samples

    @property
    def samples(self):
        return self._samples

    @property
    def gt_array(self):
        return self._gts

    @property
    def shape(self):
        return self._gts.shape

    @property
    def num_vars(self):
        return self._gts.shape[0]

    @property
    def num_samples(self):
        return self._gts.shape[1]

    @property
    def ploidy(self):
        return self._gts.shape[2]

    def get_vars(self, index):
        gts = self.gt_array[index, :, :]

        gts.flags.writeable = False
        return self.__class__(gt_array=gts, samples=self.samples)

    def get_samples(self, samples: Sequence[str] | Sequence[int]) -> Self:
        if not isinstance(samples, SequenceABC):
            raise ValueError("samples must be a sequence")
        index = numpy.where(numpy.isin(self.samples, samples))[0]
        gts = self.gt_array[:, index, :]
        samples = self.samples[index]

        samples.flags.writeable = False
        gts.flags.writeable = False
        return self.__class__(gt_array=gts, samples=samples)


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
        gts: Genotypes,
        vars_info: pandas.DataFrame | None = None,
        alleles: pandas.DataFrame | None = None,
    ):
        if not isinstance(gts, Genotypes):
            raise ValueError("gts must be a Genotypes object")
        if vars_info is not None:
            if vars_info.shape[0] != gts.num_vars:
                raise ValueError(
                    "variants_info must have the same number of rows as gts"
                )
            vars_info = _normalize_pandas_types(vars_info)

        if alleles is not None:
            if alleles.shape[0] != gts.num_vars:
                raise ValueError("alleles must have the same number of rows as gts")

        self._arrays = {"gts": gts, "vars_info": vars_info, "alleles": alleles}
        self._gt_array = gts
        self._vars_info = vars_info
        self._alleles = alleles

    @property
    def num_vars(self):
        return self._arrays["gts"].num_vars

    @property
    def num_samples(self):
        return self._arrays["gts"].num_samples

    @property
    def ploidy(self):
        return self._arrays["gts"].ploidy

    @property
    def gts(self):
        return self._arrays["gts"]


class Variants:
    def __init__(
        self, vars_chunks: Iterator[VariantsChunk], store_chunks_in_memory=False
    ):
        vars_chunks = iter(vars_chunks)
        self._first_chunk = None
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
        if self._first_chunk is not None:
            return self._first_chunk
        if self._vars_chunks is not None:
            chunk = self._vars_chunks[0]
        else:
            try:
                chunk = next(self._chunks_iter)
            except StopIteration:
                raise RuntimeError("No variants_chunks available")
            self._chunks_iter = itertools.chain([chunk], self._chunks_iter)
            self._first_chunk = chunk
        return chunk

    def iter_vars_chunks(
        self, desired_num_vars_per_chunk=DEF_NUM_VARS_PER_CHUNK
    ) -> Iterator[VariantsChunk]:
        return self._get_orig_vars_iter()

    @property
    def samples(self):
        return self._get_first_chunk().gts.samples

    @classmethod
    def from_gt_array(
        cls,
        gts: numpy.array,
        samples: list[str] | None = None,
    ) -> Self:
        chunk = VariantsChunk(gts=Genotypes(gts, samples=samples))
        return cls(vars_chunks=[chunk], store_chunks_in_memory=True)
