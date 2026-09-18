from pathlib import Path
import json

import numpy
import pandas
import pyarrow
import pyarrow.ipc

from pynei.variants import Variants, Genotypes, VariantsChunk
import pynei.config as config
from pynei.config import Compression


# 2.0 is one arrow IPC (feather v2) file, one record batch per chunk. 1.x was
# a dir with one subdir per chunk, and it is not read any more
VARS_FORMAT_VERSION = "2.0"

GTS_COL = "gts"
ALLELES_COL = "alleles"
FILE_METADATA_KEY = b"pynei"
CHUNK_METADATA_KEY = b"pynei_chunk"

# the arrow codec each member of the enum asks for, None being no compression
_ARROW_COMPRESSION = {Compression.ZSTD: "zstd", Compression.NONE: None}

_OLD_DIR_FORMAT_ERROR = (
    "{path} is a dir, so it is a vars dir of the old 1.x format, which is not "
    "read any more. Write the variants again from their VCF, or read the dir "
    "with a pynei from before the format changed."
)


def _gts_to_arrow(gts: numpy.ndarray) -> pyarrow.FixedSizeListArray:
    """The genotypes as one column, a fixed size list per variant.

    A fixed size list keeps no offsets, so the column is a single flat buffer
    of num_vars x num_samples x ploidy values in C order, which is the array
    itself. That is what lets it be read back without copying anything.
    """
    num_vars, num_samples, ploidy = gts.shape
    if gts.dtype != config.GT_NUMPY_DTYPE:
        gts = gts.astype(config.GT_NUMPY_DTYPE)
    # a chunk that comes from slicing the samples is not contiguous, and
    # reshape would not be able to give one flat buffer
    flat = pyarrow.array(numpy.ascontiguousarray(gts).reshape(-1))
    return pyarrow.FixedSizeListArray.from_arrays(flat, num_samples * ploidy)


def _arrow_to_gts(column, num_samples: int, ploidy: int) -> numpy.ndarray:
    values = column.flatten().to_numpy(zero_copy_only=False)
    return values.reshape(-1, num_samples, ploidy)


def _chunk_to_batch(chunk: VariantsChunk) -> pyarrow.RecordBatch:
    arrays = {}
    if chunk.vars_info is not None:
        table = pyarrow.Table.from_pandas(chunk.vars_info, preserve_index=False)
        for name in table.column_names:
            arrays[name] = table.column(name).combine_chunks()
    if chunk.alleles is not None:
        arrays[ALLELES_COL] = pyarrow.array(chunk.alleles)
    arrays[GTS_COL] = _gts_to_arrow(chunk.gts.gt_values)
    return pyarrow.record_batch(arrays)


def _chunk_range(vars_info) -> dict:
    """Where a chunk starts and ends, so that a region query could skip it."""
    if vars_info is None:
        return {}
    if (
        config.VAR_TABLE_CHROM_COL not in vars_info.columns
        or config.VAR_TABLE_POS_COL not in vars_info.columns
    ):
        return {}
    chroms = vars_info[config.VAR_TABLE_CHROM_COL]
    poss = vars_info[config.VAR_TABLE_POS_COL]
    return {
        "start_chrom": chroms.iloc[0],
        "start_pos": int(poss.iloc[0]),
        "end_chrom": chroms.iloc[-1],
        "end_pos": int(poss.iloc[-1]),
    }


def write_vars(
    variants: Variants,
    path: Path,
    compression: Compression = config.DEF_VARS_COMPRESSION,
):
    """It writes the variants into one vars file, to read them back fast.

    The chunks go in at the size the variants hand them out, which is the size
    a calculation asks for by default, so that reading them back does not have
    to join them and slice them again.

    ZSTD makes the file about six times smaller, NONE about five times faster
    to read. A file that has to be read in a browser wants ZSTD, because there
    it has to be downloaded and it is held in memory.
    """
    path = Path(path)
    compression = Compression(compression)
    if path.exists():
        raise ValueError(f"The path to write the variants into is already used: {path}")

    metadata = {
        "var_format_version": VARS_FORMAT_VERSION,
        "samples": list(variants.samples),
        "num_samples": variants.num_samples,
        "ploidy": variants.ploidy,
        "gts_dtype": numpy.dtype(config.GT_NUMPY_DTYPE).name,
        "missing_allele": config.MISSING_ALLELE,
    }
    write_options = pyarrow.ipc.IpcWriteOptions(
        compression=_ARROW_COMPRESSION[compression]
    )

    schema = None
    sink = None
    writer = None
    try:
        for chunk in variants.iter_vars_chunks():
            batch = _chunk_to_batch(chunk)
            if writer is None:
                metadata["num_vars_per_chunk"] = chunk.num_vars
                schema = batch.schema.with_metadata(
                    {FILE_METADATA_KEY: json.dumps(metadata).encode()}
                )
                sink = pyarrow.OSFile(str(path), "wb")
                writer = pyarrow.ipc.new_file(sink, schema, options=write_options)
            else:
                # every batch of a file shares one schema, and casting says
                # which chunk did not fit rather than failing deep in arrow
                try:
                    batch = batch.cast(schema)
                except (ValueError, pyarrow.ArrowInvalid) as error:
                    raise ValueError(
                        f"All the chunks written into a vars file must have the same "
                        f"columns and types: {error}"
                    ) from error
            chunk_metadata = json.dumps(_chunk_range(chunk.vars_info)).encode()
            writer.write_batch(
                batch, custom_metadata={CHUNK_METADATA_KEY: chunk_metadata}
            )
    finally:
        if writer is not None:
            writer.close()
        if sink is not None:
            sink.close()

    if writer is None:
        raise ValueError("There are no variants to write")


def _check_format_version(version, path):
    if version is None:
        raise ValueError(
            f"{path} has no format version, so it was not written by pynei, or it "
            f"was written by a pynei older than the one file format"
        )
    if version.split(".")[0] != VARS_FORMAT_VERSION.split(".")[0]:
        raise ValueError(
            f"{path} is a vars file of the {version} format, and this pynei reads "
            f"the {VARS_FORMAT_VERSION} one"
        )


class VariantsFile:
    """The chunks of one vars file, read one record batch at a time."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_dir():
            raise ValueError(_OLD_DIR_FORMAT_ERROR.format(path=self.path))
        # the mapping is kept open for as long as this object lives, and the
        # arrays given out keep a reference to it, so a chunk read from an
        # uncompressed file is a view on the file and nothing is copied
        self._source = pyarrow.memory_map(str(self.path), "rb")
        try:
            self._reader = pyarrow.ipc.open_file(self._source)
        except pyarrow.ArrowInvalid as error:
            raise ValueError(f"{self.path} is not a vars file: {error}") from error

        schema_metadata = self._reader.schema.metadata or {}
        raw = schema_metadata.get(FILE_METADATA_KEY)
        if raw is None:
            raise ValueError(f"{self.path} is not a vars file, it has no metadata")
        self.metadata = json.loads(raw)
        _check_format_version(self.metadata.get("var_format_version"), self.path)

        samples = self.metadata.get("samples")
        if samples is None:
            raise ValueError(f"The vars file has no samples: {self.path}")
        self.samples = tuple(samples)
        self.metadata["samples"] = self.samples
        self.num_samples = self.metadata["num_samples"]
        self.ploidy = int(self.metadata["ploidy"])
        self.num_chunks = self._reader.num_record_batches
        self._vars_info_cols = [
            name
            for name in self._reader.schema.names
            if name not in (GTS_COL, ALLELES_COL)
        ]

    def _get_metadata(self):
        # a copy, so that whoever gets it can not modify the metadata that was
        # read from the file
        return dict(self.metadata)

    def read_chunk(self, chunk_idx: int) -> VariantsChunk:
        batch = self._reader.get_batch(chunk_idx)

        gts = _arrow_to_gts(batch.column(GTS_COL), self.num_samples, self.ploidy)
        # the mask is not written, the missing alleles are already
        # MISSING_ALLELE in the values, and Genotypes asks for the two to agree
        gt_array = numpy.ma.masked_array(gts, gts == config.MISSING_ALLELE)
        chunk_kwargs = {
            "gts": Genotypes(gt_array, samples=self.samples, skip_mask_check=True)
        }

        if self._vars_info_cols:
            chunk_kwargs["vars_info"] = batch.select(self._vars_info_cols).to_pandas()
        if ALLELES_COL in batch.schema.names:
            chunk_kwargs["alleles"] = pandas.Series(
                batch.column(ALLELES_COL), dtype=config.PANDAS_ALLELES_DTYPE
            )

        return VariantsChunk(**chunk_kwargs)

    def iter_vars_chunks(self):
        for chunk_idx in range(self.num_chunks):
            yield self.read_chunk(chunk_idx)


def load_vars(vars_path: Path, desired_num_vars_per_chunk: int | None = None):
    return Variants(
        vars_chunk_iter_factory=VariantsFile(vars_path),
        desired_num_vars_per_chunk=desired_num_vars_per_chunk,
    )
