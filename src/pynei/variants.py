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
        if self.samples is None:
            raise ValueError("Cannot get samples from Genotypes without samples")

        if not isinstance(samples, SequenceABC):
            raise ValueError("samples must be a sequence")
        index = numpy.where(numpy.isin(self.samples, samples))[0]
        gts = self.gt_array[:, index, :]
        samples = self.samples[index]

        samples.flags.writeable = False
        gts.flags.writeable = False
        return self.__class__(gt_array=gts, samples=samples)


ArrayType = tuple[numpy.ndarray, pandas.DataFrame, pandas.Series, Genotypes]


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

    @property
    def vars_info(self):
        return self._arrays["vars_info"]

    @property
    def alleles(self):
        return self._arrays["alleles"]

    def get_vars(self, index):
        gts = self.gts.get_vars(index)
        if self.vars_info is not None:
            vars_info = self.vars_info.iloc[index, ...]
        else:
            vars_info = None
        if self.alleles is not None:
            alleles = self.alleles.iloc[index, ...]
        else:
            alleles = None
        return VariantsChunk(gts=gts, vars_info=vars_info, alleles=alleles)


class Variants:
    def __init__(
        self,
        vars_chunks: Iterator[VariantsChunk],
        store_chunks_in_memory=False,
        desired_num_vars_per_chunk=DEF_NUM_VARS_PER_CHUNK,
    ):
        self.desired_num_vars_per_chunk = desired_num_vars_per_chunk

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

    def iter_vars_chunks(self) -> Iterator[VariantsChunk]:
        return _resize_chunks(
            self._get_orig_vars_iter(), desired_num_rows=self.desired_num_vars_per_chunk
        )

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


def _concat_genotypes(genotypes: Sequence[Genotypes]):
    gtss = []
    for gts in genotypes:
        if gts.samples != genotypes[0].samples:
            raise ValueError("All genotypes must have the same samples")
        gtss.append(gts.gt_array)
    gts = numpy.vstack(gtss)
    return Genotypes(genotypes=gts, samples=genotypes[0].samples)


def _concatenate_arrays(arrays: list[ArrayType]) -> ArrayType:
    if isinstance(arrays[0], numpy.ndarray):
        array = numpy.vstack(arrays)
    elif isinstance(arrays[0], pandas.DataFrame):
        array = pandas.concat(arrays, axis=0)
    elif isinstance(arrays[0], pandas.Series):
        array = pandas.concat(arrays)
    elif isinstance(arrays[0], Genotypes):
        array = _concat_genotypes(arrays)
    else:
        raise ValueError("unknown type for array: " + str(type(arrays[0])))
    return array


def _concatenate_chunks(chunks: list[VariantsChunk]):
    chunks = list(chunks)

    if len(chunks) == 1:
        return chunks[0]

    arrays_to_concatenate = {"gts": [], "vars_info": [], "alleles": []}
    for chunk in chunks:
        if chunk.gts:
            arrays_to_concatenate["gts"].append(chunk.gts)
        if chunk.vars_info:
            arrays_to_concatenate["vars_info"].append(chunk.vars_info)
        if chunk.alleles:
            arrays_to_concatenate["alleles"].append(chunk.alleles)

    num_arrays = [len(arrays) for arrays in arrays_to_concatenate.values()]
    if not all([num_arrays[0] == len_ for len_ in num_arrays]):
        raise ValueError("Not all chunks have the same arrays")

    concatenated_chunk = {}
    for array_id, arrays in arrays_to_concatenate.items():
        concatenated_chunk[array_id] = _concatenate_arrays(arrays)
    concatenated_chunk = VariantsChunk(**concatenated_chunk)
    return concatenated_chunk


def _get_num_rows_in_chunk(buffered_chunk):
    if not buffered_chunk:
        return 0
    else:
        return buffered_chunk.num_vars


def _fill_buffer(buffered_chunk, chunks, desired_num_rows):
    num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
    if num_rows_in_buffer >= desired_num_rows:
        return buffered_chunk, False

    chunks_to_concat = []
    if num_rows_in_buffer:
        chunks_to_concat.append(buffered_chunk)

    total_num_rows = num_rows_in_buffer
    no_chunks_remaining = True
    for chunk in chunks:
        total_num_rows += chunk.num_vars
        chunks_to_concat.append(chunk)
        if total_num_rows >= desired_num_rows:
            no_chunks_remaining = False
            break

    if not chunks_to_concat:
        buffered_chunk = None
    elif len(chunks_to_concat) > 1:
        buffered_chunk = _concatenate_chunks(chunks_to_concat)
    else:
        buffered_chunk = chunks_to_concat[0]
    return buffered_chunk, no_chunks_remaining


def _yield_chunks_from_buffer(buffered_chunk, desired_num_rows):
    num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
    if num_rows_in_buffer == desired_num_rows:
        chunks_to_yield = [buffered_chunk]
        buffered_chunk = None
        return buffered_chunk, chunks_to_yield

    start_row = 0
    chunks_to_yield = []
    end_row = None
    while True:
        previous_end_row = end_row
        end_row = start_row + desired_num_rows
        if end_row <= num_rows_in_buffer:
            chunks_to_yield.append(buffered_chunk.get_vars(slice(start_row, end_row)))
        else:
            end_row = previous_end_row
            break
        start_row = end_row

    remainder = buffered_chunk.get_vars(slice(end_row, None))
    buffered_chunk = remainder
    return buffered_chunk, chunks_to_yield


def _resize_chunks(
    chunks: Iterator[VariantsChunk], desired_num_rows
) -> Iterator[VariantsChunk]:
    buffered_chunk = None

    while True:
        # fill buffer with equal or more than desired
        buffered_chunk, no_chunks_remaining = _fill_buffer(
            buffered_chunk, chunks, desired_num_rows
        )
        # yield chunks until buffer less than desired
        num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
        if not num_rows_in_buffer:
            break
        buffered_chunk, chunks_to_yield = _yield_chunks_from_buffer(
            buffered_chunk, desired_num_rows
        )
        for chunk in chunks_to_yield:
            yield chunk

        if no_chunks_remaining:
            yield buffered_chunk
            break
