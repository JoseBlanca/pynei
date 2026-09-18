from functools import partial
from typing import Iterator

import numpy
import pandas

from pynei.variants import Genotypes, VariantsChunk, Variants
from pynei.config import VAR_TABLE_CHROM_COL, VAR_TABLE_POS_COL, MISSING_ALLELE


class _ChunkIteratorFactory:
    def __init__(
        self,
        num_chroms,
        num_vars_per_chrom,
        dist_between_vars,
        create_gts_funct,
        num_samples,
        chunk_size,
    ):
        self.num_chroms = num_chroms
        self.num_vars_per_chrom = num_vars_per_chrom
        self.dist_between_vars = dist_between_vars
        self.create_gts_funct = create_gts_funct
        self.num_samples = num_samples
        self.samples = tuple(f"sample_{idx}" for idx in range(1, num_samples + 1))
        self.chunk_size = chunk_size
        self.ploidy = 2

        # the chunks are built once, here, and not on every call to
        # iter_vars_chunks, so that going over the variants twice gives the
        # same variants, and not a fresh set of random genotypes
        chrom_names = [f"chrom_{idx}" for idx in range(1, self.num_chroms + 1)]
        stop = self.num_vars_per_chrom * self.dist_between_vars + 1
        poss_in_chrom = numpy.arange(1, stop, self.dist_between_vars)
        self._num_vars = self.num_chroms * self.num_vars_per_chrom
        self._chroms = numpy.repeat(chrom_names, self.num_vars_per_chrom)
        self._poss = numpy.tile(poss_in_chrom, self.num_chroms)
        self._gt_array = self.create_gts_funct(
            num_vars=self._num_vars, num_samples=self.num_samples
        ).gt_values

    def _get_metadata(self):
        return {
            "samples": self.samples,
            "num_samples": self.num_samples,
            "ploidy": self.ploidy,
        }

    def iter_vars_chunks(self):
        for chunk_start in range(0, self._num_vars, self.chunk_size):
            chunk_stop = chunk_start + self.chunk_size
            chunk_gts = Genotypes(
                self._gt_array[chunk_start:chunk_stop, ...], samples=self.samples
            )
            vars_info = pandas.DataFrame(
                {
                    VAR_TABLE_CHROM_COL: self._chroms[chunk_start:chunk_stop],
                    VAR_TABLE_POS_COL: self._poss[chunk_start:chunk_stop],
                }
            )
            chunk = VariantsChunk(chunk_gts, vars_info=vars_info)
            yield chunk


def generate_vars(
    num_chroms,
    num_vars_per_chrom,
    dist_between_vars,
    create_gts_funct,
    num_samples,
    chunk_size,
):
    chunk_iterator_factory = _ChunkIteratorFactory(
        num_chroms,
        num_vars_per_chrom,
        dist_between_vars,
        create_gts_funct,
        num_samples,
        chunk_size,
    )
    variants = Variants(chunk_iterator_factory, desired_num_vars_per_chunk=chunk_size)
    return variants


def create_gts_funct(num_vars, num_samples, maf, ploidy=2):
    shape = (num_vars, num_samples, ploidy)
    rng = numpy.random.default_rng()
    gt_array = numpy.array(rng.uniform(size=shape) > maf, dtype=int)
    return gt_array


if __name__ == "__main__":
    variants = generate_vars(
        num_chroms=2,
        num_vars_per_chrom=10,
        dist_between_vars=1000,
        create_gts_funct=partial(create_gts_funct, maf=0.9),
        num_samples=10,
        chunk_size=10,
    )


class _FromGtListChunkIterFactory:
    def __init__(self, gts: list[numpy.array], samples=None):
        chunks = [
            VariantsChunk(
                gts=Genotypes(
                    numpy.ma.array(
                        gt, mask=gts == MISSING_ALLELE, fill_value=MISSING_ALLELE
                    ),
                    skip_mask_check=True,
                    samples=samples,
                ),
            )
            for gt in gts
        ]
        self._chunks = chunks

    def _get_metadata(self):
        first_chunk = self._chunks[0]
        return {
            "samples": first_chunk.gts.samples,
            "num_samples": first_chunk.num_samples,
            "ploidy": first_chunk.gts.ploidy,
        }

    def iter_vars_chunks(self) -> Iterator[VariantsChunk]:
        return iter(self._chunks)
