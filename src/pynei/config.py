from enum import Enum
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

PANDAS_STRING_STORAGE = (
    "python"  # when pyodide supports pyarrow, we will change this to 'pyarrow'
)
# PANDAS_STRING_STORAGE = "pyarrow"
DEF_NUM_VARS_PER_CHUNK = 10000
LINEAL = "lineal"
LOGARITHMIC = "logarithmic"
BinType = Enum("BinType", [LINEAL, LOGARITHMIC])
DEF_POP_NAME = "pop"
MIN_NUM_SAMPLES_FOR_POP_STAT = 20
MISSING_ALLELE = -1

DEF_NUMPY_GZIP_COMPRESSION_LEVEL = 4
