import gzip
import tempfile
from pathlib import Path
import math

import numpy
import pytest

from pynei.io_vcf import (
    parse_vcf,
    _guess_vcf_file_kind,
    _VCFKind,
    _parse_metadata,
    vars_from_vcf,
)

VCF_45 = b"""##fileformat=VCFv4.5
##fileDate=20090805
##source=myImputationProgramV3.1
##reference=file:///seq/references/1000GenomesPilot-NCBI36.fasta
##contig=<ID=20,length=62435964,assembly=B36,md5=f126cdf8a6e0c7f379d618ff66beb2da,species=\"Homo sapiens\",taxonomy=x>
##phasing=partial
##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of Samples With Data\">
##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">
##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">
##INFO=<ID=AA,Number=1,Type=String,Description=\"Ancestral Allele\">
##INFO=<ID=DB,Number=0,Type=Flag,Description=\"dbSNP membership, build 129\">
##INFO=<ID=H2,Number=0,Type=Flag,Description=\"HapMap2 membership\">
##FILTER=<ID=q10,Description=\"Quality below 10\">
##FILTER=<ID=s50,Description=\"Less than 50% of samples have data\">
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">
##FORMAT=<ID=HQ,Number=2,Type=Integer,Description=\"Haplotype Quality\">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA00001\tNA00002\tNA00003
20\t14370\trs6054257\tG\tA\t29\tPASS\tNS=3;DP=14;AF=0.5;DB;H2\tGT:GQ:DP:HQ\t0|0:48:1:51,51\t3|4:48:8:51,51\t5/6:43:5:.,.
20\t17330\t.\tT\tA\t.\tq10\tNS=3;DP=11;AF=0.017\tGT:GQ:DP:HQ\t.|0:49:3:58,50\t0|1:3:5:65,3\t0/0:41:3
20\t1110696\trs6040355\tA\tG,T\t67\tPASS\tNS=2;DP=10;AF=0.333,0.667;AA=T;DB\tGT:GQ:DP:HQ\t1|2:21:6:23,27\t2|1:2:0:18,2\t2/2:35:4
20\t1230237\t.\tT\t.\t47\tPASS\tNS=3;DP=13;AA=T\tGT:GQ:DP:HQ\t0|0:54:7:56,60\t0|0:48:4:51,51\t0/0:61:2
20\t1234567\tmicrosat1\tGTC\tG,GTCT\t50\tPASS\tNS=3;DP=9;AA=G\tGT:GQ:DP\t0/1:35:4\t0/2:17:2\t1/1:40:3
20\t1234567\tmicrosat1\tGTC\tG,GTCT\t50\tPASS\tNS=3;DP=9;AA=G\tGT:GQ:DP\t0/1:35:4\t0/2:17:2\t1/1:40:3"""


def test_vcf_file_type():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(VCF_45)
        tmp.flush()
        tmp_path = Path(tmp.name)
        assert _guess_vcf_file_kind(tmp_path) == _VCFKind.VCF

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(gzip.compress(VCF_45))
        tmp.flush()
        tmp_path = Path(tmp.name)
        assert _guess_vcf_file_kind(tmp_path) == _VCFKind.GzippedVCF


def test_metadata_parser():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(VCF_45)
        tmp.flush()
        tmp_path = Path(tmp.name)
        metadata = _parse_metadata(open(tmp_path, "rb"))
        assert len(metadata["samples"]) == 3
        assert numpy.array_equal(metadata["samples"], ["NA00001", "NA00002", "NA00003"])
        assert metadata["ploidy"] == 2


def test_vcf_parser():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(gzip.compress(VCF_45))
        tmp.flush()
        tmp_path = Path(tmp.name)
        res = parse_vcf(tmp_path)
        variants = list(res["variants"])
        snp = variants[0]

        assert snp["chrom"] == "20"
        assert snp["pos"] == 14370
        assert snp["alleles"] == ["G", "A"]
        assert math.isclose(snp["qual"], 29)
        assert numpy.array_equal(snp["gts"], [[0, 0], [3, 4], [5, 6]])
        assert numpy.all(snp["missing_mask"] == 0)

        snp = variants[1]
        assert numpy.array_equal(
            snp["missing_mask"], [[True, False], [False, False], [False, False]]
        )
        assert numpy.array_equal(snp["gts"], [[-1, 0], [0, 1], [0, 0]])
        assert math.isnan(snp["qual"])


def test_vars_from_vcf():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(VCF_45)
        tmp.flush()
        variants = vars_from_vcf(Path(tmp.name))
        assert variants.num_samples == 3
        assert variants.ploidy == 2
        chunk = list(variants.iter_vars_chunks())[0]
        assert chunk.num_vars == 6
        assert chunk.vars_info.loc[0, "chrom"] == "20"
        assert chunk.vars_info.loc[0, "pos"] == 14370
        gts = [
            [[0, 0], [3, 4], [5, 6]],
            [[-1, 0], [0, 1], [0, 0]],
            [[1, 2], [2, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 2], [1, 1]],
            [[0, 1], [0, 2], [1, 1]],
        ]
        assert numpy.array_equal(chunk.gts.gt_values, numpy.array(gts))


def test_vcf_samples_are_a_tuple():
    with tempfile.NamedTemporaryFile(suffix=".vcf") as tmp:
        tmp.write(VCF_45)
        tmp.flush()
        variants = vars_from_vcf(Path(tmp.name))
        assert variants.samples == ("NA00001", "NA00002", "NA00003")
        assert next(variants.iter_vars_chunks()).gts.samples == variants.samples


# the GT is the last field of the FORMAT, so the genotype of the last sample
# carries the end of the line with it
VCF_GT_ONLY = b"""##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3
20\t10\t.\tA\tT\t29\tPASS\t.\tGT\t0/0\t0/1\t./.
20\t20\t.\tA\tT\t29\tPASS\t.\tGT\t./.\t1/1\t0/1
20\t30\t.\tA\tT\t29\tPASS\t.\tGT:DP\t0/1:3\t0/1:3\t./.:3
"""


def test_missing_gt_in_the_last_sample():
    # the line ends after the genotype of the last sample, so its ./. comes
    # with the end of the line attached to it
    with tempfile.NamedTemporaryFile(suffix=".vcf") as tmp:
        tmp.write(VCF_GT_ONLY)
        tmp.flush()
        chunk = next(vars_from_vcf(Path(tmp.name)).iter_vars_chunks())
        expected = [
            [[0, 0], [0, 1], [-1, -1]],
            [[-1, -1], [1, 1], [0, 1]],
            [[0, 1], [0, 1], [-1, -1]],
        ]
        assert numpy.array_equal(chunk.gts.gt_values, numpy.array(expected))
        assert numpy.array_equal(chunk.gts.missing_mask, numpy.array(expected) == -1)


def test_vcf_with_windows_line_ends():
    with tempfile.NamedTemporaryFile(suffix=".vcf") as tmp:
        tmp.write(VCF_GT_ONLY.replace(b"\n", b"\r\n"))
        tmp.flush()
        chunk = next(vars_from_vcf(Path(tmp.name)).iter_vars_chunks())
        assert chunk.num_vars == 3
        assert numpy.array_equal(
            chunk.gts.gt_values[0], numpy.array([[0, 0], [0, 1], [-1, -1]])
        )


def test_an_allele_over_the_limit_is_refused():
    """A genotype is one byte, so the alleles go up to MAX_ALLELE_NUMBER. No
    real variant has 128 alleles, and paying four bytes for every genotype to
    make room for them is not worth it."""
    vcf = VCF_GT_ONLY.replace(b"./.\t1/1\t0/1", b"0/128\t1/1\t0/1")
    with tempfile.NamedTemporaryFile(suffix=".vcf") as tmp:
        tmp.write(vcf)
        tmp.flush()
        with pytest.raises(NotImplementedError, match="128"):
            next(vars_from_vcf(Path(tmp.name)).iter_vars_chunks())


def test_the_alleles_are_one_row_per_variant():
    with tempfile.NamedTemporaryFile(suffix=".vcf") as tmp:
        tmp.write(VCF_45)
        tmp.flush()
        chunk = next(vars_from_vcf(Path(tmp.name)).iter_vars_chunks())
        # the third variant has A as reference and G and T as alternatives
        assert chunk.alleles.iloc[2] == ["A", "G", "T"]
        assert chunk.alleles.iloc[0] == ["G", "A"]
