# pynei

pyNei aims to be a library to do population genetic calculations in Python using standard libraries like pandas, and numpy. One main objetive is to be compatible with [pyodide](https://pyodide.org/).

## Usage

```python
import pynei

variants = pynei.vars_from_vcf("variants.vcf.gz")
variants = pynei.filter_by_missing_data(variants, max_allowed_missing_rate=0.1)

pops = {"pop1": ["sample_1", "sample_2"], "pop2": ["sample_3", "sample_4"]}
res = pynei.calc_per_var_distribs(variants, pops=pops)
res.obs_het.mean       # one value per pop
res.maf.hist_counts    # one column per pop
```

The variants are processed chunk by chunk, so they do not have to fit in memory,
and the filters are lazy: every calculation runs them again from the source.

### Read a VCF once

Parsing a VCF is by far the most expensive thing pynei does, and every
calculation is one pass over the variants. If you are going to do more than one
thing, write the variants to a vars file once and work from it, reading a chunk
from a vars file is many times faster than parsing it from the VCF:

```python
pynei.write_vars(pynei.vars_from_vcf("variants.vcf.gz"), "variants.vars")
variants = pynei.load_vars("variants.vars")
```

### The vars file

A vars file is one file, an [arrow IPC](https://arrow.apache.org/docs/python/feather.html)
file, also called feather v2. One chunk of variants is one record batch, with
the chrom, pos, id and qual of the variants, their alleles, and their genotypes
as a fixed size list of one byte per allele. The samples, the ploidy and the
format version travel in the schema.

A genotype is one byte, so an allele goes from 0 to 127 and a missing one is
-1. Nothing else is written: the missing genotypes are already -1 in the
values, so there is no mask to keep beside them.

You choose how the genotypes are compressed when you write the file:

```python
pynei.write_vars(variants, "variants.vars", compression=pynei.Compression.NONE)
```

Over 50000 variants and 1000 samples, 100 million genotypes:

| compression | file   | write  | one pass over the file |
| ----------- | ------ | ------ | ---------------------- |
| `ZSTD`      |  18 MB | 0.14 s | 0.094 s                |
| `NONE`      | 103 MB | 0.01 s | 0.019 s                |

`ZSTD` is the default and it is the one to use in a browser, where the file has
to be downloaded first and is then held in memory. `NONE` is five times faster
to read, because the genotypes are mapped straight from the file into the
arrays without being copied, and it is worth it when the file is on a local
disk and you are going to go over it many times.

`calc_per_var_distribs` calculates several statistics in one pass, sharing the
allele counts between them, so ask it for everything you need at once instead of
calling it once per statistic.

The distances calculated with the embedding algorithm and the linkage
disequilibrium go over the variants several times by their nature.

### Using several threads

The calculations that go chunk by chunk take a `num_threads`, and they hand one
chunk at a time to each thread:

```python
res = pynei.calc_per_var_distribs(variants, num_threads=4)
```

How much it helps depends on how big the chunks are and on the python build,
because numpy lets go of the GIL while it works on a big array. The bigger the
chunk, the more of the time is spent inside numpy and the less the GIL gets in
the way. These are the speed ups of `calc_per_var_distribs` with 6 threads on 6
performance cores, over the same 400000 variants and 100 samples, changing only
the size of the chunks:

| variants per chunk | chunks | 3.14 with the GIL | 3.14 free threaded |
| ------------------ | ------ | ----------------- | ------------------ |
| 200                |   2000 | 0.9x              | 4.1x               |
| 1000               |    400 | 1.4x              | 4.4x               |
| 5000               |     80 | 3.6x              | 4.3x               |
| 20000              |     20 | 3.7x              | 3.5x               |
| 50000              |      8 | 3.0x              | 2.7x               |

With a normal python the chunks have to be big for the threads to pay, because
what is outside numpy, building the dataframes and the histograms, is done one
thread at a time. With a free threaded python they pay whatever the size. Both
of them reach the same time in the end, about 0.21 s for that dataset, and both
of them lose when there are so few chunks that the threads run out of work.

### How big a chunk is

A chunk is sized by the genotypes it holds, variants times samples, and not by
the variants, because that is what its memory and its work depend on. A fixed
number of variants means a small chunk for a few samples and a huge one for
many: 10000 variants is 2 MB of genotypes for 100 samples and 2 GB for 10000.

So the number of variants per chunk is worked out from the number of samples,
about 5 million genotypes per chunk, between a minimum and a maximum:

| samples | variants per chunk | what decides it  |
| ------- | ------------------ | ---------------- |
| 10      | 10000              | the maximum      |
| 100     | 10000              | the maximum      |
| 1000    | 5000               | the genotypes    |
| 10000   | 500                | the genotypes    |
| 100000  | 100                | the minimum      |

The maximum is there because with few samples the genotypes alone would ask for
tens of thousands of variants, and then a dataset of 50000 variants would be one
or two chunks, with nothing to share between the threads. The minimum is there
because with very many samples it would ask for so few variants that every chunk
would carry its own overhead for almost nothing.

Over 100000 variants with 6 threads, against the fixed 10000 variants per chunk
that pynei used before:

| samples | before           | now            |
| ------- | ---------------- | -------------- |
| 100     | 0.06 s, 173 MB   | 0.06 s, 171 MB |
| 1000    | 0.46 s, 729 MB   | 0.43 s, 429 MB |
| 10000   | 4.64 s, 5911 MB  | 3.63 s, 431 MB |

Give `desired_num_vars_per_chunk` to `vars_from_vcf`, to `load_vars` or to the
`Variants` itself to say the size yourself.

`create_012_gt_matrix` behaves like the table above, it is numpy all the way.
`calc_pairwise_kosman_dists` is the exception, it compares the samples pair by
pair in python, so with a normal python build the threads make it slower, 0.8x,
and it only pays on a free threaded one, 2.3x.
