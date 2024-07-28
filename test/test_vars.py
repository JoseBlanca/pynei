import numpy
import pandas
import pytest

from pynei.variants import VariantsChunk, Variants
from pynei.config import CHROM_VARIANTS_COL, POS_VARIANTS_COL


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


def test_variants_from_gts():
    num_vars = 3
    num_samples = 4
    samples = ["a", "b", "c", "d"]
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    variants = Variants.from_gt_array(gt_array, samples=samples)
    assert variants.samples == samples
    assert numpy.array_equal(next(variants.iter_vars_chunks()).gts, gt_array)

    chunk = VariantsChunk(gts=gt_array, samples=samples)
    variants = Variants(vars_chunks=[chunk])
    assert variants.samples == samples
    assert numpy.array_equal(next(variants.iter_vars_chunks()).gts, gt_array)


def test_chunk_size():
    num_vars = 100
    num_samples = 3
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    chunk = VariantsChunk(gts=gt_array)
    variants = Variants(vars_chunks=[chunk], store_chunks_in_memory=True)
    chunks = list(variants.iter_vars_chunks(num_vars_per_chunk=10))
    print(chunks[0].num_vars)
    print(len(chunks))

    chunks = list(variants.iter_vars_chunks(num_vars_per_chunk=100))
    print(chunks[0].num_vars)
    print(len(chunks))

    chunks = list(variants.iter_vars_chunks(num_vars_per_chunk=200))
    print(chunks[0].num_vars)
    print(len(chunks))
