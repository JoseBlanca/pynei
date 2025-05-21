# This is much more slower than the python one

from tempfile import NamedTemporaryFile
from pathlib import Path
import gzip
import itertools
import io

import polars
import numpy

n_threads = 2
N_ROWS_PER_CHUNK = 5000
N_ROWS_PER_CHUNK = 200
GT_MISSING_VALUE = -1
VCF_COLS = ("CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT")
BASE_SCHEMA = polars.Schema()
BASE_SCHEMA["CHROM"] = polars.Utf8
BASE_SCHEMA["POS"] = polars.Int32
BASE_SCHEMA["ID"] = polars.Utf8
BASE_SCHEMA["REF"] = polars.Utf8
BASE_SCHEMA["ALT"] = polars.Utf8
BASE_SCHEMA["QUAL"] = polars.Float32
BASE_SCHEMA["FILTER"] = polars.Utf8
BASE_SCHEMA["INFO"] = polars.Utf8
BASE_SCHEMA["FORMAT"] = polars.Utf8


VCF = """##
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA00001\tNA00002\tNA00003
20\t14370\trs6054257\tG\tA\t29\tPASS\tNS=3;DP=14;AF=0.5;DB;H2\tGT:GQ:DP:HQ\t1|2:48:1:51,51\t3|4:48:8:51,51\t5/6000:43:5:.,.
20\t17330\t.\tT\tA\t.\tq10\tNS=3;DP=11;AF=0.017\tGT:GQ:DP:HQ\t.|0:49:3:58,50\t0|1:3:5:65,3\t0/0:41:3
20\t1110696\trs6040355\tA\tG,T\t67\tPASS\tNS=2;DP=10;AF=0.333,0.667;AA=T;DB\tGT:GQ:DP:HQ\t1|2:21:6:23,27\t2|1:2:0:18,2\t2/2:35:4
20\t1230237\t.\tT\t.\t47\tPASS\tNS=3;DP=13;AA=T\tGT:GQ:DP:HQ\t0|0:54:7:56,60\t0|0:48:4:51,51\t0/0:61:2
20\t1234567\tmicrosat1\tGTC\tG,GTCT\t50\tPASS\tNS=3;DP=9;AA=G\tGT:GQ:DP\t0/1:35:4\t0/2:17:2\t1/1:40:3
20\t1234567\tmicrosat1\tGTC\tG,GTCT\t50\tPASS\tNS=3;DP=9;AA=G\tGT:GQ:DP\t0/1:35:4\t0/2:17:2\t.:40:3"""


def read_header(fhand):
    fhand.seek(0)
    while True:
        pos = fhand.tell()  # get position before reading the line
        line = fhand.readline()  # read one line (keeps buffer in sync)
        pos2 = fhand.tell()
        if not line:
            raise RuntimeError("Header line starting with '#CHROM' not found.")
        if line.startswith("#CHROM"):
            fields = line.rstrip().lstrip("#").split("\t")
            samples = fields[len(VCF_COLS) :]
            return {
                "header_pos": pos,
                "samples": samples,
                "first_line_pos": pos2,
                "header_line": line,
            }


def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def _parse_vcf_chunk(fhand, schema, columns, samples, ploidy: int):
    table = polars.read_csv(
        fhand, columns=columns, schema=schema, separator="\t", null_values="."
    )
    table = table.with_columns(
        gt_pos=polars.col("FORMAT")
        .str.split(by=":")
        .list.eval(polars.element().eq("GT"))
        .list.arg_max()
    )
    gt_idx = table.get_column("gt_pos")
    gt_fields = table.select(samples).with_columns(
        [
            polars.col(col).str.split(":").list.get(gt_idx).alias(f"{col}_GT")
            for col in samples
        ],
        include_existing=False,
    )
    gts = gt_fields.select(
        [
            polars.col(f"{s}_GT")
            .str.replace("|", "/", literal=True)
            .str.split("/")
            .list.eval(
                polars.element()
                .str.replace(".", str(GT_MISSING_VALUE), literal=True)
                .cast(polars.Int32)
                .extend_constant(GT_MISSING_VALUE, ploidy)
                .limit(ploidy)
            )
            .list.to_array(ploidy)
            .alias(s)
            for s in samples
        ]
    )
    return {"gts": gts}


def parse_vcf(vcf_fhand):
    res = read_header(vcf_fhand)
    ploidy = 2

    samples = res["samples"]
    schema = BASE_SCHEMA.copy()
    for sample in samples:
        schema[sample] = polars.Utf8
    columns = ["CHROM", "POS", "REF", "ALT", "QUAL", "FORMAT"] + samples

    vcf_fhand.seek(res["first_line_pos"])
    header_line = res["header_line"]

    np_gt_chunks = {}

    processed_vars = 0
    for chunk in batched(vcf_fhand, N_ROWS_PER_CHUNK):
        chunk_shape = (len(chunk), len(samples), ploidy)
        print("hola")
        processed_vars += len(chunk)
        try:
            np_gt_chunk = np_gt_chunks[chunk_shape]
        except KeyError:
            np_gt_chunk = numpy.empty(chunk_shape, dtype=numpy.int32)
            np_gt_chunks[chunk_shape] = np_gt_chunk
        chunk.insert(0, header_line)
        print("ready to parse")
        fhand = io.StringIO("".join(chunk))
        del chunk
        # import gc

        # gc.collect()
        res = _parse_vcf_chunk(
            fhand,
            ploidy=ploidy,
            schema=schema,
            columns=columns,
            samples=samples,
        )
        print("parsed")
        for sample_idx, sample_gts in enumerate(res["gts"].iter_columns()):
            np_gt_chunk[:, sample_idx, :] = sample_gts
        print("processed", processed_vars)
    print(processed_vars)


tmp_fhand = NamedTemporaryFile(suffix=".csv", mode="wt")
tmp_fhand.write(VCF)
tmp_fhand.flush()
fhand = open(tmp_fhand.name, "rt")

parse_vcf(fhand)
