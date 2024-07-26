from typing import Iterator, Self

import pandas
import numpy

DEF_NUM_VARIANTS = 10000


class VariantsChunk:
    def __init__(
        self,
        variants_info: pandas.DataFrame,
        alleles: pandas.DataFrame,
        gts: numpy.array,
    ): ...

    @property
    def num_variants(self): ...

    @property
    def samples(self):
        # or indi_names
        ...

    @property
    def ploidy(self): ...

    @property
    def gt_array(self): ...


class Variants:
    def __init__(
        self, variants_chunks: Iterator[VariantsChunk], store_chunks_in_memory=False
    ):
        if store_chunks_in_memory:
            self._variants_chunks = list(variants_chunks)
            self._chunks_iter = None
        else:
            self._variants_chunks = None
            self._chunks_iter = variants_chunks

    def get_variants_chunks(
        self, num_variants_per_chunk=DEF_NUM_VARIANTS, keep_only_gts=False
    ) -> Iterator[VariantsChunk]:
        if self._variants_chunks is not None:
            return iter(self._variants_chunks)
        else:
            return self._chunks_iter

    @property
    def samples(self): ...

    @classmethod
    def from_gt_array(cls, gt_array: numpy.array) -> Self: ...
