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

The genotypes are compressed with zstd, and that is not something to choose.
Over 50000 variants and 1000 samples, 100 million genotypes, the file is 18 MB
where it would be 103 MB uncompressed, and reading it takes 0.094 s where the
uncompressed one takes 0.019 s. That five times slower reading does not show in
a calculation, because the chunks are read one ahead in a thread of their own
and the reading happens while the work is being done. Over 100000 variants and
1000 samples:

|              | 1 thread | 2      | 4      | 6      |
| ------------ | -------- | ------ | ------ | ------ |
| zstd         | 1.549 s  | 0.805  | 0.438  | 0.353  |
| uncompressed | 1.543 s  | 0.792  | 0.425  | 0.344  |

They only come apart once there are more threads working than one reader can
feed, which is at about six of them, and pynei is made to run on a personal
computer. A browser wants the small file anyway, there it has to be downloaded
first and is then held in memory.

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

While one chunk is being worked on the next one is read, in a thread of its
own. Reading a chunk is decompressing it in arrow or parsing a VCF, and both of
those let go of the GIL, so the reading hides behind the work: it takes about
6% off a pass over a zstd file, and it is what makes `ZSTD` cost nothing
against `NONE`. It holds one chunk more in memory. Only one thread reads, so
this hides the reading, it does not make the reading itself faster.

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

`create_012_gt_matrix` behaves like the table above, it is numpy all the way,
and so does `calc_pairwise_kosman_dists`, which works out every pair of samples
at once with matrix products: 20 chunks of 500 variants and 1000 samples take
0.172 s with one thread and 0.051 s with six, 3.4x. It used to compare the
samples pair by pair in python, and then the threads made it slower, 0.8x.

### GWAS

`calc_gwas` tests the association of every variant with a trait, a
continuous one with a linear model and a binomial one, 0 or 1, with a
logistic one. The phenotype is a pandas Series indexed by sample name, the
covariates a DataFrame indexed the same way, and the samples without a
phenotype are left out:

```python
kinship = pynei.calc_kinship(pruned_variants)
res = pynei.calc_gwas(
    variants, phenotype, trait="continuous", covariates=covariates, kinship=kinship
)
res.stats                         # one row per variant: beta, se, p_value
res.null_model.heritability
```

The population structure is accounted for with a kinship, the genomic
relationship matrix of VanRaden and GCTA that `calc_kinship` calculates in
one pass, given as the covariance of a random polygenic effect, which is what
a structured or related panel needs. Without a kinship the top principal
components can be given as covariates, `kinship.principal_components(10)`
gives them from the kinship without going over the variants again, which is
enough for unrelated samples. Both at once is the Q+K model.

A GWAS is two passes over the variants with one job in memory between them.
The kinship is a samples x samples matrix accumulated chunk by chunk, like
the Kosman distances. The null model, covariates and kinship, is fitted once,
an eigendecomposition of the kinship for the linear mixed model and a
penalized quasi likelihood for the logistic one, and then every variant is
tested in one more pass, with the variance components kept at the null, P3D.
Every test is one matrix product per chunk, so the four models cost about the
same. Over 100000 variants:

| samples | kinship 1 thread / 6 | linear      | linear mixed | logistic    | logistic mixed |
| ------- | -------------------- | ----------- | ------------ | ----------- | -------------- |
| 100     | 0.09 s / 0.03        | 0.06 / 0.07 | 0.06 / 0.13  | 0.42 / 0.14 | 0.05 / 0.03    |
| 1000    | 0.96 s / 0.30        | 0.40 / 0.15 | 0.72 / 0.42  | 2.52 / 0.82 | 0.76 / 0.52    |

With 5000 samples the null model is what costs: 8 s of the 8.9 s of a linear
mixed model over 20000 variants, 6 s of them the eigendecomposition, and 25 s
for the logistic one, whose penalized quasi likelihood inverts a samples x
samples matrix a few dozen times. plink2 does the plain linear regression of
the 1000 samples in 0.10 s, because it keeps a genotype in 2 bits and never
turns it into a float; GMMAT does the mixed models in 1.6 s and 2.2 s.
The test of a variant with a kinship is quadratic in the samples, and
`use_grammar_gamma_approx=True` makes it linear, but it is only accurate when
the structure is weak: with three subpops at an fst of 0.3 the statistic it
gives is between half and one and a half times the exact one, so it is off by
default.

The results are checked against plink2, GMMAT, rrBLUP and R's glm on a
simulated panel, `test/gwas_reference`: the kinship matches plink2
`--make-rel`, with and without missing genotypes, the linear and the logistic
regressions match plink2 `--glm`, the linear mixed model matches rrBLUP and,
with `test="score"`, GMMAT, and the logistic mixed model matches GMMAT, to
the digits those programs print.
