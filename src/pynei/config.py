from enum import Enum

import pandas

MISSING_ALLELE = -1
MIN_NUM_GENOTYPES_FOR_POP_STAT = 20
DEF_POLY_THRESHOLD = 0.95
CHROM_VARIANTS_COL = "chrom"
POS_VARIANTS_COL = "pos"
PANDAS_FLOAT_DTYPE = pandas.Float32Dtype
PANDAS_INT_DTYPE = pandas.Int32Dtype
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
