import itertools

import numpy

from .var_generators import create_sample_names
import pandas
import pytest

from pynei.variants import Genotypes, VariantsChunk, Variants
from pynei.config import VAR_TABLE_CHROM_COL, VAR_TABLE_POS_COL


def test_gts():
    num_vars = 3
    num_samples = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    gt_array = numpy.ma.array(gt_array)
    gts = Genotypes(gt_array, samples=create_sample_names(gt_array))
    assert gts.num_vars == num_vars
    assert gts.num_samples == num_samples
    assert gts.ploidy == ploidy
    assert numpy.array_equal(gts.gt_values, gt_array)
    assert gts.samples == create_sample_names(gt_array)

    gts = Genotypes(gt_array, samples=["a", "b", "c", "d"])
    vars_slice = [1, 2]
    gts2 = gts.get_vars(vars_slice)
    assert numpy.array_equal(gts2.gt_values, gt_array[vars_slice, :, :])

    gts2 = gts.filter_samples(["b", "d"])
    assert numpy.array_equal(gts2.gt_values, gt_array[:, [1, 3], :])

    with pytest.raises(ValueError):
        gts = Genotypes(gt_array, samples=["a", "a", "c", "d"])

    with pytest.raises(ValueError):
        gts = Genotypes(gt_array, samples=["a", "b"])


def test_gts_to_012():
    gt_array = numpy.array([[[0, -1], [0, 0], [0, 0], [0, 0], [0, 1], [1, 0], [1, 1]]])
    gts = Genotypes(gt_array, samples=create_sample_names(gt_array))
    assert numpy.all(gts.to_012() == [[-1, 0, 0, 0, 1, 1, 2]])


def test_chunk():
    num_vars = 3
    num_samples = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    chunk = VariantsChunk(
        gts=Genotypes(
            numpy.ma.array(gt_array),
            samples=create_sample_names(numpy.ma.array(gt_array)),
        )
    )
    assert chunk.num_vars == num_vars
    assert chunk.num_samples == num_samples
    assert chunk.ploidy == ploidy


def test_chunk_different_num_rows():
    num_vars = 3
    num_samples = 4
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    gts = Genotypes(
        numpy.ma.array(gt_array), samples=create_sample_names(numpy.ma.array(gt_array))
    )
    vars_info = pandas.DataFrame(
        {
            VAR_TABLE_CHROM_COL: ["chr1", "chr1", "chr2"],
            VAR_TABLE_POS_COL: [1, 2, 3],
        }
    )
    chunk = VariantsChunk(gts=gts, vars_info=vars_info)
    assert chunk.num_vars == num_vars

    variants_info2 = pandas.DataFrame(
        {VAR_TABLE_CHROM_COL: ["chr1", "chr1"], VAR_TABLE_POS_COL: [1, 2]}
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
    assert numpy.array_equal(next(variants.iter_vars_chunks()).gts.gt_values, gt_array)

    variants = Variants.from_gt_array(gt_array, samples=samples)
    assert all(numpy.equal(variants.samples, samples))
    assert numpy.array_equal(next(variants.iter_vars_chunks()).gts.gt_values, gt_array)

    assert variants.num_samples == 4
    assert variants.ploidy == 2


def test_chunk_size():
    num_vars = 100
    num_samples = 3
    ploidy = 2
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    gt_array = numpy.ma.array(gt_array)
    variants = Variants.from_gt_array(gt_array, samples=create_sample_names(gt_array))
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
    gt_array = numpy.ma.array(gt_array)
    variants = Variants.from_gt_array(gt_array, samples=create_sample_names(gt_array))
    variants.desired_num_vars_per_chunk = 10
    chunks = list(variants.iter_vars_chunks())
    assert [chunk.num_vars for chunk in chunks] == [10, 5]


def test_gts_to_012_alleles_not_starting_at_zero():
    # the major allele is 2, and it is neither the allele 0 nor the first one
    gt_array = numpy.array([[[2, 2], [2, 2], [2, 1], [1, 1], [2, -1]]])
    gts = Genotypes(gt_array, samples=create_sample_names(gt_array))
    assert numpy.all(gts.to_012() == [[0, 0, 1, 2, -1]])


def test_gts_to_012_only_missing():
    gt_array = numpy.full((2, 3, 2), -1)
    gts = Genotypes(gt_array, samples=create_sample_names(gt_array))
    assert numpy.all(gts.to_012() == numpy.full((2, 3), -1))


def test_a_masked_array_says_which_alleles_are_missing():
    # a masked array can be given, and whatever it masks is taken as missing
    # however the values under the mask were left, here a 0
    gt_array = numpy.array([[[0, 0], [1, 1], [1, 1], [2, 2], [1, 1]]])
    mask = numpy.zeros_like(gt_array, dtype=bool)
    mask[0, 0, :] = True
    gts = Genotypes(numpy.ma.array(gt_array, mask=mask), samples=list("abcde"))
    # the mask is not kept, it is written into the values as MISSING_ALLELE
    assert numpy.array_equal(gts.gt_values[0, 0], [-1, -1])
    assert numpy.array_equal(gts.missing_mask, mask)
    # the major allele is 1, sample a is missing and sample d is 2/2
    assert numpy.all(gts.to_012() == [[-1, 0, 0, 2, 0]])


def test_samples_are_always_a_tuple_of_python_objects():
    gt_array = numpy.random.randint(0, 2, size=(3, 4, 2))
    expected = ("a", "b", "c", "d")

    for given_samples in (
        ["a", "b", "c", "d"],
        ("a", "b", "c", "d"),
        numpy.array(["a", "b", "c", "d"]),
    ):
        gts = Genotypes(gt_array, samples=given_samples)
        assert gts.samples == expected
        # not numpy scalars, so they can be written to json
        assert all(type(sample) is str for sample in gts.samples)

        variants = Variants.from_gt_array(gt_array, samples=given_samples)
        assert variants.samples == expected
        assert Variants.from_vars(variants).samples == expected

    # the names keep their type, only the container is normalized
    assert Variants.from_gt_array(gt_array, samples=[0, 1, 2, 3]).samples == (
        0,
        1,
        2,
        3,
    )

    # the samples are required, they are not guessed
    with pytest.raises(TypeError):
        Genotypes(gt_array)
    with pytest.raises(TypeError):
        Variants.from_gt_array(gt_array)
    with pytest.raises(ValueError, match="samples are required"):
        Genotypes(gt_array, samples=None)
    with pytest.raises(ValueError, match="samples are required"):
        Variants.from_gt_array(gt_array, samples=None)


def test_genotypes_filter_samples_gives_a_tuple():
    gt_array = numpy.random.randint(0, 2, size=(3, 4, 2))
    gts = Genotypes(gt_array, samples=numpy.array(["a", "b", "c", "d"]))

    assert gts.filter_samples(["b", "d"]).samples == ("b", "d")
    assert gts.filter_samples_with_idxs(numpy.array([0, 2])).samples == ("a", "c")
    assert gts.filter_samples_with_idxs(slice(0, 2)).samples == ("a", "b")


def test_duplicated_samples_are_refused():
    gt_array = numpy.random.randint(0, 2, size=(3, 4, 2))
    with pytest.raises(ValueError, match="Duplicated sample names"):
        Genotypes(gt_array, samples=["a", "b", "a", "b"])


def _create_chunk(chroms, poss, alleles=None, num_samples=4, ploidy=2):
    num_vars = len(poss)
    vars_info = pandas.DataFrame(
        {
            VAR_TABLE_CHROM_COL: pandas.Series(chroms, dtype=pandas.StringDtype()),
            VAR_TABLE_POS_COL: pandas.Series(poss, dtype=pandas.Int32Dtype()),
        }
    )
    gt_array = numpy.ma.array(
        numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    )
    gts = Genotypes(gt_array, samples=create_sample_names(gt_array))
    if alleles is not None:
        alleles = pandas.DataFrame(alleles, dtype=pandas.StringDtype())
    return VariantsChunk(gts=gts, vars_info=vars_info, alleles=alleles)


def test_the_tables_of_a_chunk_are_indexed_from_zero():
    """However the chunks were split and joined again, every chunk has to have
    its tables indexed from 0 to num_vars - 1, or the row labels of a
    concatenated chunk repeat themselves and .loc gives several rows."""

    class Factory:
        def _get_metadata(self):
            return {
                "samples": create_sample_names(numpy.zeros((1, 4, 2))),
                "num_samples": 4,
                "ploidy": 2,
            }

        def iter_vars_chunks(self):
            yield _create_chunk(["c1"] * 3, [1, 2, 3], alleles=[["A", "T"]] * 3)
            yield _create_chunk(["c1"] * 3, [4, 5, 6], alleles=[["A", "G"]] * 3)

    variants = Variants(Factory(), desired_num_vars_per_chunk=6)
    chunk = next(variants.iter_vars_chunks())
    assert chunk.num_vars == 6
    assert list(chunk.vars_info.index) == list(range(6))
    assert list(chunk.alleles.index) == list(range(6))
    assert chunk.vars_info.loc[4, VAR_TABLE_POS_COL] == 5

    # and the same when a chunk is split into smaller ones
    variants = Variants(Factory(), desired_num_vars_per_chunk=2)
    for chunk in variants.iter_vars_chunks():
        assert list(chunk.vars_info.index) == list(range(chunk.num_vars))
        assert list(chunk.alleles.index) == list(range(chunk.num_vars))
