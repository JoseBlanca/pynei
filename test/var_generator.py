from functools import partial

import numpy
import pandas

from pynei.variants import Genotypes, VariantsChunk, Variants
from pynei.config import VAR_TABLE_CHROM_COL, VAR_TABLE_POS_COL


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
        self.chunk_size = chunk_size

    def iter_vars_chunks(self):
        chrom_names = [f"chrom_{idx}" for idx in range(1, self.num_chroms + 1)]
        stop = self.num_vars_per_chrom * self.dist_between_vars + 1
        poss_in_chrom = numpy.arange(1, stop, self.dist_between_vars)
        num_vars = self.num_chroms * self.num_vars_per_chrom
        chroms = numpy.repeat(chrom_names, self.num_vars_per_chrom)
        poss = numpy.tile(poss_in_chrom, self.num_chroms)
        gt_array = self.create_gts_funct(
            num_vars=num_vars, num_samples=self.num_samples
        ).gt_values
        sample_names = [f"sample_{idx}" for idx in range(1, self.num_samples + 1)]

        for chunk_start in range(0, num_vars, self.chunk_size):
            chunk_stop = chunk_start + self.chunk_size
            chunk_gts = Genotypes(
                gt_array[chunk_start:chunk_stop, ...], samples=sample_names
            )
            vars_info = pandas.DataFrame(
                {
                    VAR_TABLE_CHROM_COL: chroms[chunk_start:chunk_stop],
                    VAR_TABLE_POS_COL: poss[chunk_start:chunk_stop],
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
    vars = Variants(chunk_iterator_factory, desired_num_vars_per_chunk=chunk_size)
    return vars


def create_gts_funct(num_vars, num_samples, maf, ploidy=2):
    shape = (num_vars, num_samples, ploidy)
    rng = numpy.random.default_rng()
    gt_array = numpy.array(rng.uniform(size=shape) > maf, dtype=int)
    return gt_array


if __name__ == "__main__":
    vars = generate_vars(
        num_chroms=2,
        num_vars_per_chrom=10,
        dist_between_vars=1000,
        create_gts_funct=partial(create_gts_funct, maf=0.9),
        num_samples=10,
        chunk_size=10,
    )
