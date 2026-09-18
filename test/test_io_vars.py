import tempfile
from pathlib import Path

import pytest
import pandas
import numpy

from .var_generators import create_sample_names

from pynei.variants import VariantsChunk, Variants, Genotypes
import pynei.config as config
from pynei.io_vars import write_vars, load_vars, VariantsDir
from pynei.io_vcf import vars_from_vcf
from .test_vcf import VCF_45

# the variants dir uses parquet for the variants info and the alleles
pytest.importorskip("pyarrow")


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
        gts = Genotypes(
            numpy.ma.array(gts), samples=create_sample_names(numpy.ma.array(gts))
        )
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
    variants = Variants(chunk_factory)

    with tempfile.TemporaryDirectory(suffix=".variants") as tempdir:
        write_vars(variants, tempdir)
        vars_dir = VariantsDir(tempdir)
        assert vars_dir.num_samples == 10
        variants = Variants(vars_dir)
        chunk = next(variants.iter_vars_chunks())
        assert chunk.gts.num_samples == 10
        numpy.array_equal(chunk.gts.gt_ma_array, orig_chunk.gts.gt_ma_array)
        assert chunk.vars_info.equals(orig_chunk.vars_info)


def test_vars_io_keeps_the_alleles():
    chroms = ["chrom1", "chrom1", "chrom2"]
    poss = [1, 2, 3]
    chunk_factory = _ChunkFactory(chroms, poss, num_samples=4, ploidy=2)
    alleles = pandas.DataFrame(
        [["A", "T", None], ["C", None, None], ["G", "A", "TT"]],
        dtype=config.PANDAS_STR_DTYPE(),
    )
    orig_chunk = VariantsChunk(
        gts=chunk_factory.chunk.gts,
        vars_info=chunk_factory.chunk.vars_info,
        alleles=alleles,
    )
    chunk_factory.chunk = orig_chunk
    variants = Variants(chunk_factory)

    with tempfile.TemporaryDirectory(suffix=".variants") as tempdir:
        write_vars(variants, tempdir)
        chunk = next(load_vars(tempdir).iter_vars_chunks())
        assert chunk.alleles is not None
        assert chunk.alleles.equals(orig_chunk.alleles)


def test_vcf_to_vars_dir_round_trip():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        vcf_path = tempdir / "variants.vcf"
        vcf_path.write_bytes(VCF_45)
        orig_chunk = next(vars_from_vcf(vcf_path).iter_vars_chunks())
        assert orig_chunk.alleles is not None

        vars_dir = tempdir / "vars_dir"
        write_vars(vars_from_vcf(vcf_path), vars_dir)
        variants = load_vars(vars_dir)
        chunk = next(variants.iter_vars_chunks())

        assert list(variants.samples) == list(orig_chunk.gts.samples)
        assert chunk.alleles.equals(orig_chunk.alleles)
        assert chunk.vars_info.equals(orig_chunk.vars_info)
        assert numpy.array_equal(chunk.gts.gt_ma_array, orig_chunk.gts.gt_ma_array)
        assert numpy.array_equal(chunk.gts.missing_mask, orig_chunk.gts.missing_mask)


def test_write_vars_creates_the_dir_and_refuses_a_used_one():
    chunk_factory = _ChunkFactory(["chrom1"], [1], num_samples=4, ploidy=2)
    variants = Variants(chunk_factory)

    with tempfile.TemporaryDirectory() as tempdir:
        vars_dir = Path(tempdir) / "not_created_yet" / "variants"
        write_vars(variants, vars_dir)
        assert load_vars(vars_dir).num_samples == 4

        # writing again into the same dir would mix the two sets of chunks
        with pytest.raises(ValueError):
            write_vars(variants, vars_dir)


def test_loaded_samples_are_a_tuple_and_the_metadata_is_not_aliased():
    chunk_factory = _ChunkFactory(["chrom1", "chrom1"], [1, 2], num_samples=4, ploidy=2)
    chunk_factory.chunk = VariantsChunk(
        gts=Genotypes(
            chunk_factory.chunk.gts.gt_ma_array, samples=["a", "b", "c", "d"]
        ),
        vars_info=chunk_factory.chunk.vars_info,
    )
    chunk_factory.num_samples = 4
    variants = Variants(chunk_factory)

    with tempfile.TemporaryDirectory(suffix=".variants") as tempdir:
        write_vars(variants, tempdir)
        vars_dir = VariantsDir(tempdir)

        loaded = load_vars(tempdir)
        assert loaded.samples == ("a", "b", "c", "d")
        assert next(loaded.iter_vars_chunks()).gts.samples == ("a", "b", "c", "d")

        # whoever gets the metadata can not change the one of the dir
        metadata = vars_dir._get_metadata()
        metadata["samples"] = ("z",)
        assert vars_dir._get_metadata()["samples"] == ("a", "b", "c", "d")


def test_samples_with_a_numpy_array_can_be_written():
    gt_array = numpy.random.randint(0, 2, (3, 4, 2))
    variants = Variants.from_gt_array(
        gt_array, samples=numpy.array(["a", "b", "c", "d"])
    )
    with tempfile.TemporaryDirectory(suffix=".variants") as tempdir:
        write_vars(variants, tempdir)
        assert load_vars(tempdir).samples == ("a", "b", "c", "d")
