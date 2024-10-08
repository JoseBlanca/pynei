import gzip
import tempfile
from pathlib import Path
import math

import numpy

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
20\t14370\trs6054257\tG\tA\t29\tPASS\tNS=3;DP=14;AF=0.5;DB;H2\tGT:GQ:DP:HQ\t1|2:48:1:51,51\t3|4:48:8:51,51\t5/6000:43:5:.,.
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
        vars = list(res["vars"])
        snp = vars[0]

        assert snp["chrom"] == "20"
        assert snp["pos"] == 14370
        assert snp["alleles"] == ["G", "A"]
        assert math.isclose(snp["qual"], 29)
        assert numpy.array_equal(snp["gts"], [[1, 2], [3, 4], [5, 6000]])
        assert numpy.all(snp["missing_mask"] == 0)

        snp = vars[1]
        assert numpy.array_equal(
            snp["missing_mask"], [[True, False], [False, False], [False, False]]
        )
        assert numpy.array_equal(snp["gts"], [[-1, 0], [0, 1], [0, 0]])
        assert math.isnan(snp["qual"])


def test_vars_from_vcf():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(VCF_45)
        tmp.flush()
        vars = vars_from_vcf(Path(tmp.name))
        assert vars.num_samples == 3
        assert vars.ploidy == 2
        chunk = list(vars.iter_vars_chunks())[0]
        assert chunk.num_vars == 6
        assert chunk.vars_info.loc[0, "chrom"] == "20"
        assert chunk.vars_info.loc[0, "pos"] == 14370
        gts = [
            [[1, 2], [3, 4], [5, 6000]],
            [[-1, 0], [0, 1], [0, 0]],
            [[1, 2], [2, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 2], [1, 1]],
            [[0, 1], [0, 2], [1, 1]],
        ]
        assert numpy.array_equal(chunk.gts.gt_values, numpy.array(gts))
