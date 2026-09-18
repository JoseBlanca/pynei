from enum import StrEnum
import array

import pandas
import numpy

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

PYTHON_ARRAY_TYPE = "i"
BYTE_SIZE_OF_INT = array.array(PYTHON_ARRAY_TYPE, [0]).itemsize
MAX_ALLELE_NUMBER = {1: 127, 2: 32767, 4: 2147483647}[BYTE_SIZE_OF_INT]
GT_NUMPY_DTYPE = {2: numpy.int16, 4: numpy.int32}[BYTE_SIZE_OF_INT]
# The 012 gts only hold the number of non major alleles of a genotype, 0 to the
# ploidy, or MISSING_ALLELE, so the smallest int is enough for any real ploidy
GT_012_NUMPY_DTYPE = numpy.int8

PANDAS_STRING_STORAGE = "pyarrow"
DEF_NUM_VARS_PER_CHUNK = 10000
LINEAL = "lineal"
LOGARITHMIC = "logarithmic"
# a StrEnum, so that a member is equal to the string it was built from and
# BinType(a_string) gives the member back
BinType = StrEnum("BinType", {LINEAL: LINEAL, LOGARITHMIC: LOGARITHMIC})
DEF_POP_NAME = "pop"
MIN_NUM_SAMPLES_FOR_POP_STAT = 20
MISSING_ALLELE = -1

DEF_NUMPY_GZIP_COMPRESSION_LEVEL = 4

# How many variant chunks one thread takes at a time. It is one because a
# variant chunk is already a big unit of work, thousands of variants, so
# handing out several of them at once only leaves the other threads idle:
# with 10 chunks and 50 of them per thread, one thread did everything
MAP_REDUCE_CHUNK_SIZE = 1
