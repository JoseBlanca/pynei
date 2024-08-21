from pathlib import Path
import array
from enum import Enum
import gzip
import functools

import numpy

VCF_SAMPLE_LINE_ITEMS = [
    "#CHROM",
    "POS",
    "ID",
    "REF",
    "ALT",
    "QUAL",
    "FILTER",
    "INFO",
    "FORMAT",
]


class _VCFKind(Enum):
    VCF = "vcf"
    GzippedVCF = "GzippedVCF"


def _guess_vcf_file_kind(path: Path):
    is_gzipped = False
    with path.open("rb") as fhand:
        start = fhand.read(2)
        if start[:1] == b"#":
            return _VCFKind.VCF
        elif start[:2] == b"\x1f\x8b":
            is_gzipped = True

    if not is_gzipped:
        raise ValueError(
            "Invalid VCF file, it does not start with # and its not gzipped"
        )

    with gzip.open(path) as fhand:
        start = fhand.read(1)
        if start[:1] == b"#":
            return _VCFKind.GzippedVCF
    raise ValueError("Invalid VCF gzipped file, it does not start with #")


def _parse_metadata(fhand):
    metadata = {}
    for line in fhand:
        if line.startswith(b"##"):
            pass
        elif line.startswith(b"#CHROM"):
            items = line.decode().strip().split("\t")
            if items[:9] != VCF_SAMPLE_LINE_ITEMS:
                raise ValueError(
                    "Invalid VCF file, it has an invalid sample line: {line.decode()}"
                )
            metadata["samples"] = items[9:]
            break
        else:
            raise ValueError("Invalid VCF file, it has no header")

    num_samples = len(metadata["samples"])
    try:
        var_line = next(fhand)
    except StopIteration:
        raise ValueError("Empty VCF file, it has no variants")
    var_ = _parse_var_line(var_line, num_samples, ploidy=None)
    metadata["ploidy"] = var_["gts"].shape[1]

    return metadata


def _open_vcf(fpath):
    kind = _guess_vcf_file_kind(fpath)
    if kind == _VCFKind.GzippedVCF:
        fhand = gzip.open(fpath, mode="rb")
    else:
        fhand = fpath.open("rb")
    return fhand


@functools.lru_cache
def _parse_allele(allele):
    if allele == b".":
        return True, 0
    else:
        allele = int(allele)
    if allele > 255:
        raise NotImplementedError(f"Only alleles up to 255 are implemented: {allele}")
    return False, allele


_EXPECT_PHASED = False


@functools.lru_cache
def _parse_gt(gt):
    if _EXPECT_PHASED:
        sep, other_sep = b"|", b"/"
        is_phased = True
    else:
        sep, other_sep = b"/", b"|"
        is_phased = False
    try:
        return is_phased, tuple(map(_parse_allele, gt.split(sep)))
    except ValueError:
        pass
    is_phased = not is_phased
    return is_phased, tuple(map(_parse_allele, gt.split(other_sep)))


@functools.lru_cache
def _decode_chrom(chrom):
    return chrom.decode()


@functools.lru_cache
def _get_gt_fmt_idx(gt_fmt):
    return gt_fmt.split(b":").index(b"GT")


def _parse_var_line(line, num_samples, ploidy=None):
    fields = line.split(b"\t")
    ref = fields[3].decode()
    alt = fields[4].decode().split(",")
    alleles = [ref] + alt

    gt_fmt_idx = _get_gt_fmt_idx(fields[8])

    if ploidy is None:
        alleles = _parse_gt(fields[9].split(b":")[gt_fmt_idx])
        ploidy = len(alleles)

    ref_gt_str = b"/".join([b"0"] * ploidy)
    size_of_int = 1
    gts = array.array("b", bytearray(num_samples * ploidy) * size_of_int)
    missing_mask = array.array("b", bytearray(num_samples * ploidy) * size_of_int)
    sample_idx = 0
    for gt_str in fields[9:]:
        gt_str = gt_str.split(b":")[gt_fmt_idx]
        if gt_str == ref_gt_str:
            continue
        for allele_idx, (is_missing, allele) in enumerate(_parse_gt(gt_str)[1]):
            if is_missing:
                missing_mask[sample_idx + allele_idx] = 1
            gts[sample_idx + allele_idx] = allele
        sample_idx += ploidy
    gts = numpy.frombuffer(gts, dtype=numpy.int8).reshape(num_samples, ploidy)
    missing_mask = (
        numpy.frombuffer(missing_mask, dtype=numpy.int8)
        .reshape(num_samples, ploidy)
        .astype(bool)
    )

    return {
        "chrom": _decode_chrom(fields[0]),
        "pos": int(fields[1]),
        "alleles": alleles,
        "id": fields[2].decode(),
        "qual": float(fields[5]),
        "gts": gts,
        "missing_mask": missing_mask,
    }


def _read_vars(fhand, metadata):
    for line in fhand:
        if line.startswith(b"#CHROM"):
            break

    num_samples = len(metadata["samples"])
    parse_var_line = functools.partial(
        _parse_var_line, num_samples=num_samples, ploidy=metadata["ploidy"]
    )
    vars = map(parse_var_line, fhand)
    return vars


def parse_vcf(fpath: Path):
    fpath = Path(fpath)
    fhand = _open_vcf(fpath)
    metadata = _parse_metadata(fhand)

    fhand = _open_vcf(fpath)
    vars = _read_vars(fhand, metadata)

    # print(numpy.array([[1, 2], [3, 4], [5, 6]]).tobytes())
    return {"metadata": metadata, "vars": vars}
