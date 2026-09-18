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
thing, write the variants to a vars dir once and work from it, reading a chunk
from a vars dir is many times faster than parsing it from the VCF:

```python
pynei.write_vars(pynei.vars_from_vcf("variants.vcf.gz"), "variants.vars")
variants = pynei.load_vars("variants.vars")
```

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

The default of 10000 variants per chunk is already in the good range. A dataset
small enough to be a couple of chunks is not worth threading at all.

`create_012_gt_matrix` behaves like the table above, it is numpy all the way.
`calc_pairwise_kosman_dists` is the exception, it compares the samples pair by
pair in python, so with a normal python build the threads make it slower, 0.8x,
and it only pays on a free threaded one, 2.3x.
