from enum import StrEnum
import array

import pandas
import numpy
import pyarrow

MIN_NUM_GENOTYPES_FOR_POP_STAT = 20
DEF_POLY_THRESHOLD = 0.95

VAR_TABLE_CHROM_COL = "chrom"
VAR_TABLE_POS_COL = "pos"
VAR_TABLE_QUAL_COL = "qual"
VAR_TABLE_ID_COL = "id"

PANDAS_FLOAT_DTYPE = pandas.Float32Dtype
PANDAS_INT_DTYPE = pandas.Int32Dtype
PANDAS_POS_DTYPE = pandas.UInt64Dtype
PANDAS_STR_DTYPE = pandas.StringDtype

# A genotype is one byte. An allele goes from 0 to MAX_ALLELE_NUMBER and a
# missing one is MISSING_ALLELE, so an int8 holds 128 alleles, far more than
# any real variant has. It used to be an int32, which cost four times the
# memory and four times the file to make room for alleles that nobody has.
PYTHON_ARRAY_TYPE = "b"
BYTE_SIZE_OF_GT = array.array(PYTHON_ARRAY_TYPE, [0]).itemsize
GT_NUMPY_DTYPE = numpy.int8
MAX_ALLELE_NUMBER = int(numpy.iinfo(GT_NUMPY_DTYPE).max)
# The 012 gts only hold the number of non major alleles of a genotype, 0 to the
# ploidy, or MISSING_ALLELE, so the smallest int is enough for any real ploidy
GT_012_NUMPY_DTYPE = numpy.int8

PANDAS_STRING_STORAGE = "pyarrow"
# One row of the alleles table is the alleles of one variant, ["A", "T", "G"],
# and not one column per allele index, because every chunk of a vars file
# shares one schema however many alleles its variants happen to have
PANDAS_ALLELES_DTYPE = pandas.ArrowDtype(pyarrow.list_(pyarrow.large_string()))
# How big a chunk is, counted in genotypes, variants x samples, and not in
# variants, because that is what the memory and the work of a chunk depend on.
# A fixed number of variants means a small chunk for a few samples and a huge
# one for many: 10000 variants is 2 MB of gts for 100 samples and 2 GB for
# 10000. 5 million genotypes is about 100 MB of gts, and up to there the peak
# memory hardly moves, it is what the calculations allocate on top that shows.
DEF_NUM_GTS_PER_CHUNK = 5_000_000
# with few samples the genotype budget would ask for so many variants that a
# normal dataset would be two or three chunks, and then there is nothing to
# share between the threads. It also bounds what the vars info of a chunk costs
MAX_NUM_VARS_PER_CHUNK = 10_000
# and with very many samples it would ask for so few variants that every chunk
# would carry its own python overhead for almost nothing
MIN_NUM_VARS_PER_CHUNK = 100
LINEAL = "lineal"
LOGARITHMIC = "logarithmic"
# a StrEnum, so that a member is equal to the string it was built from and
# BinType(a_string) gives the member back
BinType = StrEnum("BinType", {LINEAL: LINEAL, LOGARITHMIC: LOGARITHMIC})
DEF_POP_NAME = "pop"
MIN_NUM_SAMPLES_FOR_POP_STAT = 20
MISSING_ALLELE = -1

# The genotypes of a vars file are always compressed with zstd. Not
# compressing them makes them about five times faster to read, but the chunks
# are read one ahead in a thread of their own, and that hides the reading
# behind the work: the two of them only differ once there are more threads
# working than one reader can feed, about six of them, and pynei is made to
# run on a personal computer. A browser needs the small file anyway, there the
# file is downloaded and then held in memory
VARS_COMPRESSION = "zstd"

# How many chunks are read ahead of the work, in a thread of their own.
# Reading a chunk is decompressing it in arrow or parsing a VCF, and both of
# them let go of the GIL, so the next chunk can be read while the one in hand
# is worked on. It costs the memory of the chunks read ahead, and one is
# enough: it hides the reading behind the work, and a second one would only
# wait. 0 turns it off
NUM_CHUNKS_READ_AHEAD = 1
# How many variant chunks one thread takes at a time. It is one because a
# variant chunk is already a big unit of work, thousands of variants, so
# handing out several of them at once only leaves the other threads idle:
# with 10 chunks and 50 of them per thread, one thread did everything
MAP_REDUCE_CHUNK_SIZE = 1
