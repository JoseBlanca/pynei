import numpy
import pandas
import pytest

from pynei.variants import VariantsChunk
from pynei.config import CHROM_VARIANTS_COL, POS_VARIANTS_COL, PANDAS_STRING_STORAGE


def test_chunk():
    num_vars = 3
    num_samples = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    chunk = VariantsChunk(gts=gt_array)
    assert chunk.num_vars == num_vars
    assert chunk.num_samples == num_samples
    assert chunk.ploidy == ploidy


def test_chunk_different_num_rows():
    num_vars = 3
    num_samples = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    variants_info = pandas.DataFrame(
        {
            CHROM_VARIANTS_COL: ["chr1", "chr1", "chr2"],
            POS_VARIANTS_COL: [1, 2, 3],
        }
    )
    chunk = VariantsChunk(gts=gt_array, variants_info=variants_info)
    assert chunk.num_vars == num_vars

    variants_info2 = pandas.DataFrame(
        {CHROM_VARIANTS_COL: ["chr1", "chr1"], POS_VARIANTS_COL: [1, 2]}
    )
    with pytest.raises(ValueError):
        VariantsChunk(gts=gt_array, variants_info=variants_info2)

    alleles = pandas.DataFrame([["A", "T", ""], ["C", "G", ""], ["A", "C", "T"]])
    VariantsChunk(gts=gt_array, alleles=alleles)

    with pytest.raises(ValueError):
        VariantsChunk(gts=gt_array, alleles=alleles.iloc[:1, :])

    VariantsChunk(gts=gt_array, samples=["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        VariantsChunk(gts=gt_array, samples=["a", "b"])
