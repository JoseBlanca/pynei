import numpy
import pandas
import pytest

from pynei.variants import Genotypes, VariantsChunk, Variants
from pynei.config import CHROM_VARIANTS_COL, POS_VARIANTS_COL


def test_gts():
    num_vars = 3
    num_samples = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    gts = Genotypes(gt_array)
    assert gts.num_vars == num_vars
    assert gts.num_samples == num_samples
    assert gts.ploidy == ploidy
    assert numpy.array_equal(gts.gt_array, gt_array)
    assert gts.samples is None

    gts = Genotypes(gt_array, samples=["a", "b", "c", "d"])
    vars_slice = [1, 2]
    gts2 = gts.get_vars(vars_slice)
    assert numpy.array_equal(gts2.gt_array, gt_array[vars_slice, :, :])

    gts2 = gts.get_samples(["b", "d"])
    assert numpy.array_equal(gts2.gt_array, gt_array[:, [1, 3], :])

    with pytest.raises(ValueError):
        gts = Genotypes(gt_array, samples=["a", "a", "c", "d"])

    with pytest.raises(ValueError):
        gts = Genotypes(gt_array, samples=["a", "b"])


def test_chunk():
    num_vars = 3
    num_samples = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    chunk = VariantsChunk(gts=Genotypes(gt_array))
    assert chunk.num_vars == num_vars
    assert chunk.num_samples == num_samples
    assert chunk.ploidy == ploidy


def test_chunk_different_num_rows():
    num_vars = 3
    num_samples = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    gts = Genotypes(gt_array)
    vars_info = pandas.DataFrame(
        {
            CHROM_VARIANTS_COL: ["chr1", "chr1", "chr2"],
            POS_VARIANTS_COL: [1, 2, 3],
        }
    )
    chunk = VariantsChunk(gts=gts, vars_info=vars_info)
    assert chunk.num_vars == num_vars

    variants_info2 = pandas.DataFrame(
        {CHROM_VARIANTS_COL: ["chr1", "chr1"], POS_VARIANTS_COL: [1, 2]}
    )
    with pytest.raises(ValueError):
        VariantsChunk(gts=gts, vars_info=variants_info2)

    alleles = pandas.DataFrame([["A", "T", ""], ["C", "G", ""], ["A", "C", "T"]])
    VariantsChunk(gts=gts, alleles=alleles)

    with pytest.raises(ValueError):
        VariantsChunk(gts=gts, alleles=alleles.iloc[:1, :])


def test_variants_from_gts():
    num_vars = 3
    num_samples = 4
    samples = ["a", "b", "c", "d"]
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    variants = Variants.from_gt_array(gt_array, samples=samples)
    assert all(numpy.equal(variants.samples, samples))
    assert numpy.array_equal(next(variants.iter_vars_chunks()).gts.gt_array, gt_array)

    gts = Genotypes(gt_array, samples=samples)
    chunk = VariantsChunk(gts=gts)
    variants = Variants(vars_chunks=[chunk])
    assert all(numpy.equal(variants.samples, samples))
    assert numpy.array_equal(next(variants.iter_vars_chunks()).gts.gt_array, gt_array)


def test_chunk_size():
    num_vars = 100
    num_samples = 3
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    chunk = VariantsChunk(gts=Genotypes(gt_array))
    variants = Variants(vars_chunks=[chunk], store_chunks_in_memory=True)
    variants.desired_num_vars_per_chunk = 10
    chunks = list(variants.iter_vars_chunks())
    assert [chunk.num_vars for chunk in chunks] == [10] * 10

    variants.desired_num_vars_per_chunk = 100
    chunks = list(variants.iter_vars_chunks())
    assert [chunk.num_vars for chunk in chunks] == [100]

    variants.desired_num_vars_per_chunk = 200
    chunks = list(variants.iter_vars_chunks())
    assert [chunk.num_vars for chunk in chunks] == [100]

    num_vars = 15
    num_samples = 3
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    chunk = VariantsChunk(gts=Genotypes(gt_array))
    variants = Variants(vars_chunks=[chunk], store_chunks_in_memory=True)
    variants.desired_num_vars_per_chunk = 10
    chunks = list(variants.iter_vars_chunks())
    assert [chunk.num_vars for chunk in chunks] == [10, 5]
