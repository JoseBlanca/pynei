from typing import Iterator

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
    def __init__(self, variant_chunks: Iterator[VariantsChunk]): ...

    def get_variants_chunks(
        self, num_variants_per_chunk=DEF_NUM_VARIANTS, keep_only_gts=False
    ) -> Iterator[VariantsChunk]: ...

    @property
    def num_variants(self): ...

    @property
    def samples(self): ...
