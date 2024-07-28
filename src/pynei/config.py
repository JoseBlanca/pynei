import pandas

MISSING_ALLELE = -1
DEFAULT_NAME_POP_ALL_INDIS = "all_indis"
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
DEF_NUM_VARIANTS_PER_CHUNK = 10000
