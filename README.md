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

How much it helps depends on the python build. These are the speed ups measured
on 6 performance cores, over 100000 variants and 100 samples:

|                          | 3.14 with the GIL | 3.14 free threaded |
| ------------------------ | ----------------- | ------------------ |
| `calc_per_var_distribs`  | 3.2x              | 3.4x               |
| `create_012_gt_matrix`   | 4.4x              | 4.5x               |
| `calc_pairwise_kosman_dists` | 0.8x          | 2.3x               |

The Kosman distances are the exception: they compare the samples pair by pair in
python, which holds the GIL, so with a normal python build asking for threads
makes them slower, not faster. Leave them with one thread unless you are running
a free threaded python.

The chunks have to be big enough for the threads to have something to do. With
the default of 10000 variants per chunk a dataset of 100000 variants is 10
chunks, which is enough for a few threads, but a much smaller dataset is not
worth threading.
