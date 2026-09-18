"""It makes the dataset and the reference numbers that test_gwas.py checks against.

It simulates a structured population, three subpops, writes it as a vars
file and as a VCF, simulates a continuous and a binomial trait with two
covariates, and then runs plink2 and R (GMMAT, rrBLUP and glm) on them. What
it writes into this dir is what the tests read, so they do not need plink2
or R. Run it from the root of the project:

    uv run python test/gwas_reference/make_reference.py

It needs plink2 and Rscript in the PATH, and the R packages GMMAT and rrBLUP.
"""

from pathlib import Path
import gzip
import shutil
import subprocess
import sys

import numpy
import pandas

from pynei import Variants, write_vars

REF_DIR = Path(__file__).parent
WORK_DIR = REF_DIR / "work"

SEED = 42
NUM_SAMPLES = 200
NUM_VARS = 1200
NUM_CHROMS = 2
NUM_POPS = 3
FAMILY_SIZE = 4
FST = 0.1
HERITABILITY = 0.5
NUM_CAUSAL = 5
CAUSAL_EFFECT = 0.6
MISSING_RATE = 0.03


def simulate_genotypes(rng):
    """Families of full sibs within three subpops.

    The families are what gives the kinship its off diagonal structure. With
    unrelated samples and only 1200 markers the bulk of the kinship is close
    to an identity matrix, and then the genetic and the residual variances
    are not identifiable: the REML fit runs to a boundary with the residual
    variance at zero, in GMMAT and in pynei alike.
    """
    num_families = NUM_SAMPLES // FAMILY_SIZE
    family_pops = rng.integers(0, NUM_POPS, size=num_families)
    p_anc = rng.uniform(0.1, 0.9, size=NUM_VARS)
    a = p_anc * (1 - FST) / FST
    b = (1 - p_anc) * (1 - FST) / FST
    p_pop = numpy.stack([rng.beta(a, b) for _ in range(NUM_POPS)])
    alleles = numpy.empty((NUM_VARS, NUM_SAMPLES, 2), dtype=numpy.int8)
    pops = numpy.repeat(family_pops, FAMILY_SIZE)
    for family_idx, pop in enumerate(family_pops):
        # two parents, vars x 2 alleles each, and every child takes one
        # allele from each parent at random
        parents = (
            rng.uniform(size=(2, NUM_VARS, 2)) < p_pop[pop][None, :, None]
        ).astype(numpy.int8)
        for child_idx in range(FAMILY_SIZE):
            sample_idx = family_idx * FAMILY_SIZE + child_idx
            for parent_idx in range(2):
                picked = rng.integers(0, 2, size=NUM_VARS)
                alleles[:, sample_idx, parent_idx] = parents[
                    parent_idx, numpy.arange(NUM_VARS), picked
                ]
    return alleles, pops


def simulate_traits(rng, dosages, pops):
    # dosages: samples x vars, alt allele count
    p = dosages.mean(axis=0) / 2
    poly = (p > 0.05) & (p < 0.95)
    z = (dosages[:, poly] - 2 * p[poly]) / numpy.sqrt(2 * p[poly] * (1 - p[poly]))
    poly_effects = rng.standard_normal(z.shape[1]) * numpy.sqrt(
        HERITABILITY / z.shape[1]
    )
    genetic = z @ poly_effects
    causal = rng.choice(numpy.flatnonzero(poly), NUM_CAUSAL, replace=False)
    genetic = genetic + dosages[:, causal] @ numpy.full(NUM_CAUSAL, CAUSAL_EFFECT)
    cov1 = rng.standard_normal(NUM_SAMPLES)
    cov2 = rng.integers(0, 2, size=NUM_SAMPLES)
    # the pops differ in their mean, so the structure confounds the trait
    fixed = 0.5 * cov1 + 0.8 * cov2 + 0.7 * pops
    noise = rng.standard_normal(NUM_SAMPLES) * numpy.sqrt(1 - HERITABILITY)
    cont = fixed + genetic + noise
    liability = (
        fixed
        + genetic
        + rng.standard_normal(NUM_SAMPLES) * numpy.sqrt(1 - HERITABILITY)
    )
    binom = (liability > numpy.quantile(liability, 0.6)).astype(int)
    return cont, binom, cov1, cov2, causal


def write_vcf(path, alleles, samples, chroms, poss, ids):
    with open(path, "w") as fhand:
        fhand.write("##fileformat=VCFv4.2\n")
        for chrom in dict.fromkeys(chroms):
            fhand.write(f"##contig=<ID={chrom}>\n")
        fhand.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fhand.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(samples)
            + "\n"
        )
        for var_idx in range(alleles.shape[0]):
            gts = "\t".join(
                "./." if a < 0 else f"{a}/{b}" for a, b in alleles[var_idx].tolist()
            )
            fhand.write(
                f"{chroms[var_idx]}\t{poss[var_idx]}\t{ids[var_idx]}\tA\tT\t.\t.\t.\tGT\t{gts}\n"
            )


def run(cmd, cwd):
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    rng = numpy.random.default_rng(SEED)
    alleles, pops = simulate_genotypes(rng)
    samples = [f"s{idx:03d}" for idx in range(NUM_SAMPLES)]
    vars_per_chrom = NUM_VARS // NUM_CHROMS
    chroms = [
        f"chr{idx + 1}" for idx in range(NUM_CHROMS) for _ in range(vars_per_chrom)
    ]
    poss = [
        1000 * (idx + 1) for _ in range(NUM_CHROMS) for idx in range(vars_per_chrom)
    ]
    ids = [f"var{idx:04d}" for idx in range(NUM_VARS)]
    dosages = alleles.sum(axis=2).T
    cont, binom, cov1, cov2, causal = simulate_traits(rng, dosages, pops)

    WORK_DIR.mkdir(exist_ok=True)
    vars_info = pandas.DataFrame({"chrom": chroms, "pos": poss, "id": ids})
    variants = Variants.from_gt_array(alleles, samples=samples, vars_info=vars_info)
    (REF_DIR / "sim.vars").unlink(missing_ok=True)
    write_vars(variants, REF_DIR / "sim.vars")
    write_vcf(WORK_DIR / "sim.vcf", alleles, samples, chroms, poss, ids)

    pheno = pandas.DataFrame(
        {
            "IID": samples,
            "cont": cont,
            "binom": binom,
            "cov1": cov1,
            "cov2": cov2,
            "pop": pops,
        }
    )
    pheno.to_csv(REF_DIR / "phenotypes.csv", index=False)
    pandas.DataFrame({"id": [ids[idx] for idx in causal]}).to_csv(
        REF_DIR / "causal_vars.csv", index=False
    )
    pheno[["IID", "cont", "binom"]].to_csv(
        WORK_DIR / "pheno.txt", sep="\t", index=False
    )
    pheno[["IID", "cov1", "cov2"]].to_csv(WORK_DIR / "covar.txt", sep="\t", index=False)
    pandas.DataFrame(dosages.T, index=ids, columns=samples).to_csv(
        WORK_DIR / "dosages.csv"
    )

    # the same genotypes with some of them missing, to check what is done
    # with a missing genotype. The kinship of the tests with them is the
    # one of the complete genotypes, so that only the tests are compared
    missing_alleles = alleles.copy()
    is_missing = rng.uniform(size=(NUM_VARS, NUM_SAMPLES)) < MISSING_RATE
    missing_alleles[is_missing] = -1
    (REF_DIR / "sim_missing.vars").unlink(missing_ok=True)
    write_vars(
        Variants.from_gt_array(missing_alleles, samples=samples, vars_info=vars_info),
        REF_DIR / "sim_missing.vars",
    )
    write_vcf(WORK_DIR / "sim_missing.vcf", missing_alleles, samples, chroms, poss, ids)

    plink2 = ["plink2", "--pheno", "pheno.txt", "--covar", "covar.txt", "--1"]
    run(
        plink2
        + ["--vcf", "sim.vcf", "--make-bed", "--make-rel", "square", "--out", "sim"],
        WORK_DIR,
    )
    run(
        plink2 + ["--vcf", "sim.vcf", "--glm", "hide-covar", "--out", "plink2"],
        WORK_DIR,
    )
    run(
        plink2
        + [
            "--vcf",
            "sim_missing.vcf",
            "--make-bed",
            "--make-rel",
            "square",
            "--out",
            "sim_missing",
        ],
        WORK_DIR,
    )
    run(["Rscript", str(REF_DIR / "reference.R")], WORK_DIR)

    # plink2 writes the relationship matrix as text with no header
    for name in ["sim", "sim_missing"]:
        with (
            open(WORK_DIR / f"{name}.rel") as src,
            gzip.open(REF_DIR / f"plink2_rel_{name}.txt.gz", "wt") as dst,
        ):
            shutil.copyfileobj(src, dst)
    for name in ["plink2.cont.glm.linear", "plink2.binom.glm.logistic.hybrid"]:
        shutil.copy(WORK_DIR / name, REF_DIR / (name + ".tsv"))
    for name in [
        "gmmat_lmm_score.tsv",
        "gmmat_glmm_score.tsv",
        "gmmat_lmm_score_missing.tsv",
        "gmmat_glmm_score_missing.tsv",
        "rrblup_lmm.tsv",
        "r_glm_score.tsv",
        "r_null_models.tsv",
    ]:
        shutil.copy(WORK_DIR / name, REF_DIR / name)


if __name__ == "__main__":
    sys.exit(main())
