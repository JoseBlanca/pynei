from array import array

import numpy

vcf_path = "../nei_rs/snps.vcf.gz"


def gt_to_int(gt):
    if gt == "./.":
        return -1, -1
    else:
        allele1, allele2 = gt.split("/")
        return int(allele1), int(allele2)


def polars_table_of_arrays_to_3d_numpy(table):
    n_rows, n_cols = table.shape
    ploidy = table.dtypes[0].shape[0]
    array = numpy.empty((n_rows, n_cols, ploidy), dtype=numpy.int16)
    for col_idx, col in enumerate(table.iter_columns()):
        array[:, col_idx, :] = col.to_numpy().flatten().reshape(n_rows, ploidy)
    return array


import functools


@functools.lru_cache
def parse_allele(allele):
    return -1 if allele == b"." else int(allele)


@functools.lru_cache
def parse_gt(gt):
    alleles = tuple(map(parse_allele, gt.split(b"/")))
    return alleles


@functools.lru_cache
def decode_chrom(chrom):
    return chrom.decode()


def parse_var_line(line):
    fields = line.split(b"\t")
    chrom = decode_chrom(fields[0])
    pos = int(fields[1])
    ref = fields[3].decode()
    alt = fields[4].decode().split(",")
    alleles = [ref] + alt

    gts = tuple(map(parse_gt, fields[9:]))

    return {"chrom": chrom, "pos": pos, "alleles": alleles, "gts": gts}


import array
import itertools

num_samples = 15504
ploidy = 2


def parse_var_line2(line):
    fields = line.split(b"\t")
    chrom = decode_chrom(fields[0])
    pos = int(fields[1])
    ref = fields[3].decode()
    alt = fields[4].decode().split(",")
    alleles = [ref] + alt

    ref_gt_str = b"/".join([b"0"] * ploidy)
    size_of_int = 4
    gts = array.array("i", bytearray(num_samples * ploidy) * size_of_int)
    for sample_idx, gt_str in enumerate(fields[9:]):
        if gt_str == ref_gt_str:
            continue
        for allele_idx, allele in enumerate(parse_gt(gt_str)):
            gts[sample_idx + allele_idx] = allele
    gts = numpy.frombuffer(gts, dtype=numpy.int32).reshape(num_samples, ploidy)

    return {"chrom": chrom, "pos": pos, "alleles": alleles, "gts": gts}


if False:
    import numpy
    import pandas

    num_header_lines = 2
    samples = pandas.read_csv(
        vcf_path, skiprows=num_header_lines, nrows=0, sep="\t", compression="gzip"
    ).columns[9:]

    cols = [
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "FORMAT",
    ] + samples.tolist()
    dtype = {"#CHROM": str, "POS": numpy.uint32}
    ploidy = 2

    for chunk in pandas.read_csv(
        vcf_path,
        sep="\t",
        header=0,
        skiprows=num_header_lines,
        chunksize=1000,
        parse_dates=False,
        compression="gzip",
        quoting=0,
        memory_map=True,
        usecols=cols,
        dtype=dtype,
    ):
        gts = chunk.loc[:, samples]
        # gts = gts.map(lambda gt: array('i', (int(allele) for allele in gt.split("/"))))
        # gts = gts.to_numpy()
        parsed_gts = numpy.zeros((gts.shape[0], gts.shape[1] * 2), dtype=numpy.int16)
        parsed_gts = numpy.zeros((gts.shape[0], gts.shape[1], 2), dtype=numpy.int16)
        for row_idx, (_, variant) in enumerate(gts.iterrows()):
            if False:
                alleles_for_variant = (
                    variant.str.split(pat="/", expand=True, n=ploidy)
                    .astype(numpy.int16)
                    .to_numpy()
                    .flatten()
                )
                parsed_gts[row_idx] = alleles_for_variant
            elif False:
                alleles_for_variant = array(
                    "i", (int(allele) for gt in variant for allele in gt.split("/"))
                )
                parsed_gts[row_idx] = alleles_for_variant
            else:
                parsed_gts[row_idx, ...] = [gt_to_int(gt) for gt in variant]

        print(parsed_gts)

elif False:
    import gzip
    import polars

    table = polars.read_csv(
        gzip.open(vcf_path, "rt"),
        skip_rows=2,
        separator="\t",
        try_parse_dates=False,
        n_threads=1,
        batch_size=1000,
        has_header=True,
    )
    table = table.with_columns(
        gt_pos=polars.col("FORMAT")
        .str.split(by=":")
        .list.eval(polars.element().eq("GT"))
        .list.arg_max()
    )
    samples = table.columns[9:]

    gts = table.select(samples).select(
        polars.col("*")
        .exclude("gt_pos")
        .str.split(":")
        .list.get("gt_pos")
        .str.split(by="/")
        .list.eval(polars.element().str.replace(".", "-1"))
        .cast(polars.List(polars.Int32))
        .list.to_array(2)
        # .list.to_struct()
    )
    gts = polars_table_of_arrays_to_3d_numpy(gts)
    print(gts)
elif True:
    from pynei.io_vcf import parse_vcf

    list(parse_vcf(vcf_path)["vars"])
elif True:
    import gzip
    import itertools
    import numpy

    num_vars_per_chunk = 1000
    n_header_lines = 3
    fhand = gzip.open(vcf_path, "rb")
    lines = (line for line in fhand)
    var_lines = itertools.dropwhile(lambda line: line.startswith(b"#"), lines)
    chunks_with_lines = itertools.batched(var_lines, num_vars_per_chunk)

    gts = None
    vars_gts = None
    for var_lines in chunks_with_lines:
        vars = map(parse_var_line2, var_lines)
        for var_idx, var in enumerate(vars):
            if gts is None:
                num_samples = num_samples
                ploidy = ploidy
                gts = numpy.zeros(
                    (num_vars_per_chunk, num_samples, ploidy), dtype=numpy.int16
                )
                gts[var_idx, ...] = var["gts"]
        print(gts)
        print(gts.shape)
elif True:
    import gzip
    import itertools
    import numpy

    num_vars_per_chunk = 1000
    n_header_lines = 3
    fhand = gzip.open(vcf_path, "rb")
    lines = (line for line in fhand)
    var_lines = itertools.dropwhile(lambda line: line.startswith(b"#"), lines)
    chunks_with_lines = itertools.batched(var_lines, num_vars_per_chunk)

    gts = None
    vars_gts = None
    for var_lines in chunks_with_lines:
        vars = map(parse_var_line, var_lines)
        for var_idx, var in enumerate(vars):
            if gts is None:
                num_samples = len(var["gts"])
                ploidy = len(var["gts"][0])
                gts = numpy.zeros(
                    (num_vars_per_chunk, num_samples, ploidy), dtype=numpy.int16
                )
            gts[var_idx, ...] = var["gts"]
        print(gts)
        print(gts.shape)
elif True:
    import gzip
    import itertools
    import numpy

    num_vars_per_chunk = 1000
    n_header_lines = 3
    fhand = gzip.open(vcf_path, "rb")
    lines = (line for line in fhand)
    var_lines = itertools.dropwhile(lambda line: line.startswith(b"#"), lines)
    chunks_with_lines = itertools.batched(var_lines, num_vars_per_chunk)

    gts = None
    vars_gts = None
    for var_lines in chunks_with_lines:
        vars = map(parse_var_line, var_lines)
        for var_idx, var in enumerate(vars):
            if gts is None:
                num_samples = len(var["gts"])
                ploidy = len(var["gts"][0])
                gts = numpy.zeros(
                    (num_vars_per_chunk, num_samples, ploidy), dtype=numpy.int16
                )
            gts[var_idx, ...] = var["gts"]
        print(gts)
        print(gts.shape)
else:
    import gzip
    import itertools
    import functools
    import numpy

    num_vars_per_chunk = 1000
    n_header_lines = 3
    fhand = gzip.open(vcf_path, "rb")
    lines = (line for line in fhand)
    var_lines = itertools.dropwhile(lambda line: line.startswith(b"#"), lines)
    chunks_with_lines = itertools.batched(var_lines, num_vars_per_chunk)

    num_samples = 15504
    ploidy = 2
    gts = numpy.zeros((num_vars_per_chunk, num_samples, ploidy), dtype=numpy.int16)
    for var_lines in chunks_with_lines:
        # vars = map(parse_var_line, var_lines)
        for var_idx, var_line in enumerate(var_lines):
            # parse_var_line = functools.partial(parse_var_line2, gts=gts, var_idx=var_idx)
            parse_var_line2(var_line, gts=gts, var_idx=var_idx)
