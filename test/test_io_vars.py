import tempfile

import pandas
import numpy

from pynei.variants import VariantsChunk, Variants, Genotypes
import pynei.config as config
from pynei.io_vars import write_vars, VariantsDir


class _ChunkFactory:
    def __init__(self, chroms, poss, num_samples, ploidy):
        chroms = pandas.Series(
            chroms,
            dtype=config.PANDAS_STR_DTYPE(),
        )
        pos = pandas.Series(poss, dtype=config.PANDAS_POS_DTYPE())
        vars_info = pandas.DataFrame({"chrom": chroms, "pos": pos})
        num_vars = pos.size
        self.num_samples = num_samples
        gts = numpy.random.randint(
            0, 2, (num_vars, num_samples, ploidy), dtype=config.GT_NUMPY_DTYPE()
        )
        gts = Genotypes(numpy.ma.array(gts))
        self.chunk = VariantsChunk(gts=gts, vars_info=vars_info)

    def iter_vars_chunks(self):
        return iter([self.chunk])

    def _get_metadata(self):
        first_chunk = self.chunk
        return {
            "samples": first_chunk.gts.samples,
            "num_samples": self.num_samples,
            "ploidy": first_chunk.gts.ploidy,
        }


def test_vars_io():
    chroms = ["chrom1", "chrom2", "chrom3", "chrom4", "chrom5"]
    poss = [1, 2, 3, 4, 5]
    num_samples = 10
    chunk_factory = _ChunkFactory(chroms, poss, num_samples=num_samples, ploidy=2)
    orig_chunk = chunk_factory.chunk
    vars = Variants(chunk_factory)

    with tempfile.TemporaryDirectory(suffix=".vars") as tempdir:
        write_vars(vars, tempdir)
        vars_dir = VariantsDir(tempdir)
        assert vars_dir.num_samples == 10
        vars = Variants(vars_dir)
        chunk = next(vars.iter_vars_chunks())
        assert chunk.gts.num_samples == 10
        numpy.array_equal(chunk.gts.gt_ma_array, orig_chunk.gts.gt_ma_array)
        assert chunk.vars_info.equals(orig_chunk.vars_info)
