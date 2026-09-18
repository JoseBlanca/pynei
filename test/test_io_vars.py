import tempfile
from pathlib import Path

import pytest
import pandas
import numpy

from .var_generators import create_sample_names

from pynei.variants import VariantsChunk, Variants, Genotypes, calc_num_vars_per_chunk
import pynei.config as config
from pynei.config import Compression
from pynei.io_vars import write_vars, load_vars, VariantsFile, VARS_FORMAT_VERSION
from pynei.io_vcf import vars_from_vcf
from .test_vcf import VCF_45

# the vars file is an arrow IPC file
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
            0, 2, (num_vars, num_samples, ploidy), dtype=config.GT_NUMPY_DTYPE
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


def _create_path(tempdir, name="variants.vars"):
    return Path(tempdir) / name


def test_vars_io():
    chroms = ["chrom1", "chrom2", "chrom3", "chrom4", "chrom5"]
    poss = [1, 2, 3, 4, 5]
    num_samples = 10
    chunk_factory = _ChunkFactory(chroms, poss, num_samples=num_samples, ploidy=2)
    orig_chunk = chunk_factory.chunk
    variants = Variants(chunk_factory)

    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        assert path.is_file()

        vars_file = VariantsFile(path)
        assert vars_file.num_samples == 10
        variants = Variants(vars_file)
        chunk = next(variants.iter_vars_chunks())
        assert chunk.gts.num_samples == 10
        assert numpy.array_equal(chunk.gts.gt_ma_array, orig_chunk.gts.gt_ma_array)
        assert chunk.vars_info.equals(orig_chunk.vars_info)


def test_the_genotypes_are_one_byte():
    chunk_factory = _ChunkFactory(["chrom1"] * 3, [1, 2, 3], num_samples=4, ploidy=2)
    variants = Variants(chunk_factory)
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        chunk = next(load_vars(path).iter_vars_chunks())
        assert chunk.gts.gt_values.dtype == numpy.int8
        assert config.MAX_ALLELE_NUMBER == 127


def test_vars_io_keeps_the_alleles():
    chroms = ["chrom1", "chrom1", "chrom2"]
    poss = [1, 2, 3]
    chunk_factory = _ChunkFactory(chroms, poss, num_samples=4, ploidy=2)
    alleles = pandas.Series(
        [["A", "T"], ["C"], ["G", "A", "TT"]], dtype=config.PANDAS_ALLELES_DTYPE
    )
    orig_chunk = VariantsChunk(
        gts=chunk_factory.chunk.gts,
        vars_info=chunk_factory.chunk.vars_info,
        alleles=alleles,
    )
    chunk_factory.chunk = orig_chunk
    variants = Variants(chunk_factory)

    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        chunk = next(load_vars(path).iter_vars_chunks())
        assert chunk.alleles is not None
        assert chunk.alleles.equals(orig_chunk.alleles)
        # one row is the alleles of one variant, so a variant with more
        # alleles than the others does not add columns to the whole chunk
        assert chunk.alleles.iloc[2] == ["G", "A", "TT"]


def test_alleles_with_different_counts_in_different_chunks():
    """The chunks of a file share one schema, and the alleles used to be one
    column per allele index, which changed from chunk to chunk."""

    def create_chunk(poss, alleles):
        factory = _ChunkFactory(["c1"] * len(poss), poss, num_samples=4, ploidy=2)
        return VariantsChunk(
            gts=factory.chunk.gts,
            vars_info=factory.chunk.vars_info,
            alleles=pandas.Series(alleles, dtype=config.PANDAS_ALLELES_DTYPE),
        )

    class Factory:
        def _get_metadata(self):
            return {
                "samples": create_sample_names(numpy.zeros((1, 4, 2))),
                "num_samples": 4,
                "ploidy": 2,
            }

        def iter_vars_chunks(self):
            yield create_chunk([1, 2], [["A", "T"], ["A", "T"]])
            yield create_chunk([3, 4], [["A", "T", "G", "C"], ["A", "G"]])

    variants = Variants(Factory(), desired_num_vars_per_chunk=2)
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        chunks = list(load_vars(path, desired_num_vars_per_chunk=2).iter_vars_chunks())
        assert chunks[0].alleles.to_list() == [["A", "T"], ["A", "T"]]
        assert chunks[1].alleles.to_list() == [["A", "T", "G", "C"], ["A", "G"]]
        # and read as one chunk they just join, nothing is padded
        chunk = next(load_vars(path, desired_num_vars_per_chunk=4).iter_vars_chunks())
        assert chunk.alleles.to_list() == [
            ["A", "T"],
            ["A", "T"],
            ["A", "T", "G", "C"],
            ["A", "G"],
        ]


@pytest.mark.parametrize("compression", [Compression.ZSTD, Compression.NONE, "zstd"])
def test_the_compression_is_chosen_when_writing(compression):
    chunk_factory = _ChunkFactory(["c1"] * 5, [1, 2, 3, 4, 5], num_samples=6, ploidy=2)
    orig_chunk = chunk_factory.chunk
    variants = Variants(chunk_factory)
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path, compression=compression)
        chunk = next(load_vars(path).iter_vars_chunks())
        assert numpy.array_equal(chunk.gts.gt_values, orig_chunk.gts.gt_values)


def test_an_uncompressed_file_is_read_without_copying_the_genotypes():
    chunk_factory = _ChunkFactory(["c1"] * 5, [1, 2, 3, 4, 5], num_samples=6, ploidy=2)
    variants = Variants(chunk_factory)
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path, compression=Compression.NONE)
        chunk = next(load_vars(path).iter_vars_chunks())
        gts = chunk.gts.gt_values
        # the array is a view on the mapped file, not a copy of it, and it is
        # read only so that nothing can write on the file through it
        assert not gts.flags.writeable
        assert not gts.flags.owndata


def test_the_chunks_are_written_at_the_size_they_are_read_at():
    num_samples = 10
    num_vars = 25
    chunk_factory = _ChunkFactory(
        ["c1"] * num_vars, list(range(num_vars)), num_samples=num_samples, ploidy=2
    )
    variants = Variants(chunk_factory, desired_num_vars_per_chunk=10)
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        vars_file = VariantsFile(path)
        assert vars_file.num_chunks == 3
        assert vars_file.metadata["num_vars_per_chunk"] == 10
        assert [chunk.num_vars for chunk in vars_file.iter_vars_chunks()] == [10, 10, 5]


def test_vcf_to_vars_file_round_trip():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        vcf_path = tempdir / "variants.vcf"
        vcf_path.write_bytes(VCF_45)
        orig_chunk = next(vars_from_vcf(vcf_path).iter_vars_chunks())
        assert orig_chunk.alleles is not None

        path = tempdir / "variants.vars"
        write_vars(vars_from_vcf(vcf_path), path)
        variants = load_vars(path)
        chunk = next(variants.iter_vars_chunks())

        assert list(variants.samples) == list(orig_chunk.gts.samples)
        assert chunk.alleles.equals(orig_chunk.alleles)
        assert chunk.vars_info.equals(orig_chunk.vars_info)
        assert numpy.array_equal(chunk.gts.gt_ma_array, orig_chunk.gts.gt_ma_array)
        assert numpy.array_equal(chunk.gts.missing_mask, orig_chunk.gts.missing_mask)
        # the mask is not in the file, it is the MISSING_ALLELE in the values
        assert numpy.array_equal(
            chunk.gts.missing_mask, chunk.gts.gt_values == config.MISSING_ALLELE
        )


def test_write_vars_refuses_a_used_path():
    chunk_factory = _ChunkFactory(["chrom1"], [1], num_samples=4, ploidy=2)
    variants = Variants(chunk_factory)

    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        assert load_vars(path).num_samples == 4

        # writing again would leave the two sets of chunks mixed
        with pytest.raises(ValueError):
            write_vars(variants, path)


def test_the_format_version_is_checked():
    chunk_factory = _ChunkFactory(["chrom1"], [1], num_samples=4, ploidy=2)
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(Variants(chunk_factory), path)
        assert VariantsFile(path).metadata["var_format_version"] == VARS_FORMAT_VERSION

        # a file written by a pynei of another format is not read silently
        import json
        import pyarrow
        import pyarrow.ipc
        from pynei.io_vars import FILE_METADATA_KEY

        reader = pyarrow.ipc.open_file(pyarrow.memory_map(str(path), "rb"))
        metadata = json.loads(reader.schema.metadata[FILE_METADATA_KEY])
        metadata["var_format_version"] = "99.0"
        schema = reader.schema.with_metadata(
            {FILE_METADATA_KEY: json.dumps(metadata).encode()}
        )
        batches = [reader.get_batch(idx) for idx in range(reader.num_record_batches)]
        other = Path(tempdir) / "other.vars"
        with pyarrow.OSFile(str(other), "wb") as sink:
            with pyarrow.ipc.new_file(sink, schema) as writer:
                for batch in batches:
                    writer.write_batch(batch.cast(schema))
        with pytest.raises(ValueError, match="99.0"):
            VariantsFile(other)


def test_an_old_vars_dir_says_what_happened():
    with tempfile.TemporaryDirectory() as tempdir:
        old_dir = Path(tempdir) / "old.vars"
        old_dir.mkdir()
        (old_dir / "var_dir_metadata.json").write_text("{}")
        with pytest.raises(ValueError, match="1.x"):
            load_vars(old_dir)


def test_a_file_that_is_not_a_vars_file_is_refused():
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        path.write_bytes(b"not an arrow file at all")
        with pytest.raises(ValueError, match="not a vars file"):
            load_vars(path)


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

    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        vars_file = VariantsFile(path)

        loaded = load_vars(path)
        assert loaded.samples == ("a", "b", "c", "d")
        assert next(loaded.iter_vars_chunks()).gts.samples == ("a", "b", "c", "d")

        # whoever gets the metadata can not change the one of the file
        metadata = vars_file._get_metadata()
        metadata["samples"] = ("z",)
        assert vars_file._get_metadata()["samples"] == ("a", "b", "c", "d")


def test_samples_with_a_numpy_array_can_be_written():
    gt_array = numpy.random.randint(0, 2, (3, 4, 2))
    variants = Variants.from_gt_array(
        gt_array, samples=numpy.array(["a", "b", "c", "d"])
    )
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        assert load_vars(path).samples == ("a", "b", "c", "d")


def test_the_written_chunk_size_is_the_one_a_calculation_asks_for():
    num_samples = 10
    num_vars = 30
    chunk_factory = _ChunkFactory(
        ["c1"] * num_vars, list(range(num_vars)), num_samples=num_samples, ploidy=2
    )
    variants = Variants(chunk_factory)
    with tempfile.TemporaryDirectory() as tempdir:
        path = _create_path(tempdir)
        write_vars(variants, path)
        vars_file = VariantsFile(path)
        # few samples and few variants, so it all fits in one chunk
        assert calc_num_vars_per_chunk(num_samples) >= num_vars
        assert vars_file.num_chunks == 1
