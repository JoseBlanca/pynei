import tempfile
import time

import numpy
import pandas
import pyarrow

# numpy and pyarrow does not like each other 
# writting numpy compressed arrays is very slow, compression with the gzip module is even slower

num_vars = 10000
num_samples = 15000
ploidy = 2
rng = numpy.random.default_rng()
gts = rng.integers(0, high=2, size=(num_vars, num_samples, ploidy), dtype=numpy.uint16)
missing = rng.integers(0, high=2, size=(num_vars, num_samples, ploidy), dtype=bool)
gts = numpy.ma.masked_array(gts, missing)

samples = numpy.array([f"sample_{i+1}" for i in range(num_samples)], dtype=str)

chroms = pandas.Series(["chrom1"] * num_vars, dtype=pandas.StringDtype("pyarrow"))
pos = pandas.Series(numpy.arange(num_vars, dtype=numpy.uint32))
vars_info = pandas.DataFrame({"chrom": chroms, "pos": pos})

with tempfile.TemporaryDirectory() as tempdir:
    vars_path = f"{tempdir}/vars_info.parquet"
    gts_path = f"{tempdir}/gts.npz"
    gts_parquet_path = f"{tempdir}/gts.parquet"

    print("Writing")
    start_time = time.time()

    numpy.savez(gts_path, gts=gts.data, masks=gts.mask)
    vars_info.to_parquet(vars_path)

    end_write_time = time.time()

    print("Reading")
    reading_start_time = time.time()
    gts, masks= numpy.load(gts_path, allow_pickle=False)
    gts = numpy.ma.masked_array(gts, masks)
    vars_info = pyarrow.parquet.read_table(vars_path).to_pandas()
    reading_end_time = time.time()

print(f"Write time: {end_write_time - start_time}")
print(f"Read time: {reading_end_time - reading_start_time}")



# TODO read VCF with pyarrow.csv.read_csv