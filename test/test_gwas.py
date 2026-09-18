"""The GWAS is checked against plink2, GMMAT, rrBLUP and R's glm.

None of them is needed to run the tests. They were run once by
gwas_reference/make_reference.py, which says how, and what they wrote is
kept in that dir and read here. The numbers of a few variants, the causal
ones and one more, are also written down here, so that a test says what it
expects. The tolerances are what the references print: plink2 writes six
significant digits, and R's glm converges to 1e-8 in the deviance.
"""

from pathlib import Path
import gzip

import numpy
import pandas
import pytest

from pynei import (
    Variants,
    calc_gwas,
    calc_kinship,
    load_vars,
    GWASModel,
    TestType,
    TraitType,
)
from pynei.gwas import _betainc, _chi2_sf_1df, _t_sf_two_sided, _LMMNull
from pynei.var_filters import filter_samples

from .var_generators import create_sample_names

REF_DIR = Path(__file__).parent / "gwas_reference"

# the causal variants of the simulation and a non causal one
SOME_VARS = ["var0000", "var0052", "var0629", "var0751", "var1137", "var1188"]

# plink2 --glm on the continuous trait, beta, se and p
PLINK2_LINEAR = {
    "var0000": (-0.424136, 0.139354, 0.00265846),
    "var0052": (-0.697724, 0.122348, 4.28981e-08),
    "var0629": (-0.813852, 0.161809, 1.10646e-06),
    "var0751": (-0.0963977, 0.130636, 0.461451),
    "var1137": (-0.171137, 0.145987, 0.242511),
    "var1188": (-0.655393, 0.138912, 4.51958e-06),
}
# plink2 --glm on the binomial trait, log odds ratio, its se and p
PLINK2_LOGISTIC = {
    "var0000": (-0.572579, 0.261917, 0.0288081),
    "var0052": (-0.852823, 0.248979, 0.000614166),
    "var0629": (-0.949571, 0.323553, 0.00333739),
    "var0751": (-0.265917, 0.219207, 0.225098),
    "var1137": (-0.427448, 0.252402, 0.0903561),
    "var1188": (-0.830184, 0.264071, 0.00166771),
}
# rrBLUP GWAS with P3D, kinship and the binary covariate, -log10(p)
RRBLUP_LMM_MINUS_LOG10_P = {
    "var0000": 0.215210,
    "var0052": 3.618699,
    "var0629": 4.339896,
    "var0751": 2.065325,
    "var1137": 1.319283,
    "var1188": 2.367279,
}
# GMMAT glmm.score, the variance of the score and p, for the linear and the
# logistic mixed models, with the complete and with the missing genotypes
GMMAT_LMM = {
    "var0000": (29.8774, 0.360526),
    "var0052": (43.8076, 0.00118985),
    "var0629": (31.7241, 4.81005e-05),
    "var0751": (44.5825, 0.00439226),
    "var1137": (43.3724, 0.013926),
    "var1188": (47.3766, 0.0010734),
}
GMMAT_LMM_MISSING = {
    "var0000": (29.9961, 0.495719),
    "var0052": (46.0763, 0.00105895),
    "var0629": (31.9307, 7.37888e-05),
    "var0751": (44.422, 0.00675644),
    "var1137": (42.1841, 0.0125321),
    "var1188": (47.9106, 0.00150577),
}
GMMAT_GLMM = {
    "var0000": (6.48664, 0.702659),
    "var0052": (8.98834, 0.0306703),
    "var0629": (6.49956, 0.0895104),
    "var0751": (10.563, 0.0142331),
    "var1137": (9.098, 0.093808),
    "var1188": (9.05095, 0.0262737),
}
GMMAT_GLMM_MISSING = {
    "var0000": (6.43005, 0.685719),
    "var0052": (8.73718, 0.0291759),
    "var0629": (6.19348, 0.12319),
    "var0751": (10.1532, 0.026766),
    "var1137": (8.92384, 0.108032),
    "var1188": (8.7063, 0.0196806),
}
# R anova(glm, test = "Rao"), the score statistic and p
R_GLM_SCORE = {
    "var0000": (4.938245, 0.026268700),
    "var0052": (12.484427, 0.000410359),
    "var0629": (9.165576, 0.002466100),
    "var0751": (1.480401, 0.223711736),
    "var1137": (2.911424, 0.087954199),
    "var1188": (10.382961, 0.001271835),
}
# GMMAT glmmkin, the variance of the kinship effect, the residual one and
# the intercept and covariate effects
GMMAT_LMM_NULL = (1.221617, 0.342359, (4.678021, 0.473361, 1.110279))
GMMAT_GLMM_NULL = (1.508057, None, (-1.416464, 0.753476, 1.583210))
# plink2 --make-rel, some entries, the samples 0 to 3 are full sibs
PLINK2_REL = {
    (0, 0): 1.09309,
    (1, 1): 1.22825,
    (0, 1): 0.648081,
    (0, 2): 0.615611,
    (0, 4): -0.0945533,
    (0, 199): -0.0760273,
    (100, 101): 0.604995,
}
PLINK2_REL_MISSING = {
    (0, 0): 1.09626,
    (0, 1): 0.650379,
    (0, 4): -0.103505,
    (100, 101): 0.604119,
}


@pytest.fixture(scope="module")
def reference():
    variants = load_vars(REF_DIR / "sim.vars")
    pheno = pandas.read_csv(REF_DIR / "phenotypes.csv", index_col="IID")
    return {
        "variants": variants,
        "variants_missing": load_vars(REF_DIR / "sim_missing.vars"),
        "pheno": pheno,
        "covariates": pheno[["cov1", "cov2"]],
        "kinship": calc_kinship(variants),
    }


def _read_plink2_rel(name):
    with gzip.open(REF_DIR / f"plink2_rel_{name}.txt.gz", "rt") as fhand:
        return numpy.loadtxt(fhand)


def _read_tsv(name):
    return pandas.read_csv(REF_DIR / name, sep="\t")


def _some_stats(res):
    return res.stats.set_index("id").loc[SOME_VARS]


def _assert_null(null, expected, tol=1e-5):
    genetic_variance, residual_variance, coefs = expected
    assert abs(null.genetic_variance - genetic_variance) < tol
    if residual_variance is None:
        assert null.residual_variance is None
    else:
        assert abs(null.residual_variance - residual_variance) < tol
    assert numpy.allclose(null.covariate_effects.to_numpy(), coefs, atol=tol)


def _assert_gmmat(res, expected, tol=1e-4):
    stats = _some_stats(res)
    for var_id, (var, p_value) in expected.items():
        assert abs(1 / stats.loc[var_id, "se"] ** 2 / var - 1) < tol
        assert abs(numpy.log10(stats.loc[var_id, "p_value"] / p_value)) < tol


def _log10_p_diff(res, ref_p):
    return numpy.nanmax(
        numpy.abs(numpy.log10(res.stats["p_value"]) - numpy.log10(ref_p))
    )


def test_kinship_matches_plink2(reference):
    kinship = reference["kinship"]
    assert kinship.num_vars == 1200
    assert kinship.samples == reference["variants"].samples
    matrix = kinship.matrix.to_numpy()
    for (row, col), value in PLINK2_REL.items():
        assert abs(matrix[row, col] - value) < 1e-5
    assert numpy.abs(matrix - _read_plink2_rel("sim")).max() < 1e-5

    # a missing genotype does not count for its pairs, as plink2 does it
    kinship = calc_kinship(reference["variants_missing"])
    matrix = kinship.matrix.to_numpy()
    for (row, col), value in PLINK2_REL_MISSING.items():
        assert abs(matrix[row, col] - value) < 1e-5
    assert numpy.abs(matrix - _read_plink2_rel("sim_missing")).max() < 1e-5


def test_kinship_of_some_samples_and_threads(reference):
    variants = reference["variants"]
    kinship = reference["kinship"]
    samples = list(variants.samples[10:50])
    some = calc_kinship(variants, samples=samples, num_threads=2)
    assert some.samples == tuple(samples)
    # the allele frequencies are those of the samples given, so it is not
    # the same as slicing the kinship of all of them
    all_sliced = kinship.filter_samples(samples)
    assert all_sliced.matrix.shape == (40, 40)
    assert not numpy.allclose(some.matrix.to_numpy(), all_sliced.matrix.to_numpy())

    pcs = kinship.principal_components(3)
    assert list(pcs.columns) == ["PC0", "PC1", "PC2"]
    assert tuple(pcs.index) == variants.samples
    # the first component tells the subpops apart
    pops = reference["pheno"]["pop"]
    assert pcs.groupby(pops)["PC0"].mean().std() > pcs["PC0"].std()

    with pytest.raises(ValueError, match="not in the kinship"):
        kinship.filter_samples(["nobody"])


def test_lm_matches_plink2(reference):
    ref = _read_tsv("plink2.cont.glm.linear.tsv")
    res = calc_gwas(
        reference["variants"],
        reference["pheno"]["cont"],
        trait="continuous",
        covariates=reference["covariates"],
    )
    assert res.null_model.model == GWASModel.LM
    assert res.test == TestType.WALD
    assert res.trait == TraitType.CONTINUOUS
    assert res.stats.shape[0] == 1200
    assert list(res.stats["id"]) == list(ref["ID"])
    stats = _some_stats(res)
    for var_id, (beta, se, p_value) in PLINK2_LINEAR.items():
        assert abs(stats.loc[var_id, "beta"] - beta) < 1e-5
        assert abs(stats.loc[var_id, "se"] - se) < 1e-5
        assert abs(stats.loc[var_id, "p_value"] / p_value - 1) < 1e-5
    # plink2 tests the minor allele, pynei the non major one, the same here
    assert numpy.abs(res.stats["allele_freq"] - ref["A1_FREQ"]).max() < 1e-6
    assert numpy.abs(res.stats["beta"] - ref["BETA"]).max() < 1e-5
    assert numpy.abs(res.stats["se"] - ref["SE"]).max() < 1e-5
    assert numpy.abs(res.stats["p_value"] / ref["P"] - 1).max() < 1e-5


def test_lmm_wald_matches_rrblup(reference):
    # rrBLUP takes its fixed effects as factors, so only the binary
    # covariate was given to it
    ref = _read_tsv("rrblup_lmm.tsv")
    res = calc_gwas(
        reference["variants"],
        reference["pheno"]["cont"],
        trait="continuous",
        covariates=reference["pheno"][["cov2"]],
        kinship=reference["kinship"],
    )
    assert res.null_model.model == GWASModel.LMM
    assert res.test == TestType.WALD
    stats = _some_stats(res)
    for var_id, minus_log10_p in RRBLUP_LMM_MINUS_LOG10_P.items():
        assert abs(-numpy.log10(stats.loc[var_id, "p_value"]) - minus_log10_p) < 1e-4
    minus_log10_p = -numpy.log10(res.stats["p_value"])
    assert numpy.abs(minus_log10_p - ref["cont"]).max() < 1e-4


def test_lmm_score_matches_gmmat(reference):
    ref = _read_tsv("gmmat_lmm_score.tsv")
    nulls = _read_tsv("r_null_models.tsv").set_index("model")
    res = calc_gwas(
        reference["variants"],
        reference["pheno"]["cont"],
        trait="continuous",
        covariates=reference["covariates"],
        kinship=reference["kinship"],
        test="score",
    )
    null = res.null_model
    _assert_null(null, GMMAT_LMM_NULL)
    assert abs(null.genetic_variance - nulls.loc["lmm", "tau"]) < 1e-5
    assert abs(null.residual_variance - nulls.loc["lmm", "sigma2"]) < 1e-5
    assert abs(null.heritability - 1.221617 / (1.221617 + 0.342359)) < 1e-5
    _assert_gmmat(res, GMMAT_LMM)
    assert _log10_p_diff(res, ref["PVAL"]) < 1e-4
    # the variance of the score is the one of GMMAT
    assert numpy.abs(1 / res.stats["se"] ** 2 / ref["VAR"] - 1).max() < 1e-5

    # with some genotypes missing GMMAT gives them the mean of the variant
    ref = _read_tsv("gmmat_lmm_score_missing.tsv")
    res = calc_gwas(
        reference["variants_missing"],
        reference["pheno"]["cont"],
        trait="continuous",
        covariates=reference["covariates"],
        kinship=reference["kinship"],
        test="score",
    )
    _assert_gmmat(res, GMMAT_LMM_MISSING)
    assert _log10_p_diff(res, ref["PVAL"]) < 1e-4


def test_glm_wald_matches_plink2(reference):
    ref = _read_tsv("plink2.binom.glm.logistic.hybrid.tsv")
    res = calc_gwas(
        reference["variants"],
        reference["pheno"]["binom"],
        trait="binomial",
        covariates=reference["covariates"],
    )
    assert res.null_model.model == GWASModel.GLM
    assert res.test == TestType.WALD
    assert res.null_model.residual_variance is None
    # plink2 falls back to a Firth regression for the variant that separates
    # the cases from the controls, pynei gives it nan
    firth = (ref["FIRTH?"] == "Y").to_numpy()
    assert firth.sum() == 1
    assert res.stats["p_value"].isna().to_numpy().tolist() == firth.tolist()
    stats = _some_stats(res)
    # plink2 stops its logistic fit earlier than pynei does
    for var_id, (log_or, se, p_value) in PLINK2_LOGISTIC.items():
        assert abs(stats.loc[var_id, "beta"] - log_or) < 1e-5
        assert abs(stats.loc[var_id, "se"] - se) < 1e-4
        assert abs(stats.loc[var_id, "p_value"] / p_value - 1) < 5e-3
    stats = res.stats[~firth]
    ref = ref[~firth]
    assert numpy.abs(stats["beta"] - numpy.log(ref["OR"])).max() < 1e-4
    assert numpy.abs(stats["se"] - ref["LOG(OR)_SE"]).max() < 1e-4
    assert numpy.abs(stats["p_value"] / ref["P"] - 1).max() < 5e-3


def test_glm_score_matches_r(reference):
    ref = _read_tsv("r_glm_score.tsv")
    res = calc_gwas(
        reference["variants"],
        reference["pheno"]["binom"],
        trait="binomial",
        covariates=reference["covariates"],
        test="score",
    )
    assert res.test == TestType.SCORE
    stats = _some_stats(res)
    for var_id, (score, p_value) in R_GLM_SCORE.items():
        assert (
            abs((stats.loc[var_id, "beta"] / stats.loc[var_id, "se"]) ** 2 - score)
            < 1e-3
        )
        assert abs(numpy.log10(stats.loc[var_id, "p_value"] / p_value)) < 1e-3
    chi2 = (res.stats["beta"] / res.stats["se"]) ** 2
    assert numpy.nanmax(numpy.abs(chi2 - ref["score"])) < 1e-2
    assert _log10_p_diff(res, ref["p"]) < 1e-3


def test_glmm_matches_gmmat(reference):
    ref = _read_tsv("gmmat_glmm_score.tsv")
    nulls = _read_tsv("r_null_models.tsv").set_index("model")
    res = calc_gwas(
        reference["variants"],
        reference["pheno"]["binom"],
        trait="binomial",
        covariates=reference["covariates"],
        kinship=reference["kinship"],
    )
    null = res.null_model
    assert null.model == GWASModel.GLMM
    assert res.test == TestType.SCORE
    _assert_null(null, GMMAT_GLMM_NULL)
    assert abs(null.genetic_variance - nulls.loc["glmm", "tau"]) < 1e-5
    _assert_gmmat(res, GMMAT_GLMM)
    assert _log10_p_diff(res, ref["PVAL"]) < 1e-4

    ref = _read_tsv("gmmat_glmm_score_missing.tsv")
    res = calc_gwas(
        reference["variants_missing"],
        reference["pheno"]["binom"],
        trait="binomial",
        covariates=reference["covariates"],
        kinship=reference["kinship"],
    )
    _assert_gmmat(res, GMMAT_GLMM_MISSING)
    assert _log10_p_diff(res, ref["PVAL"]) < 1e-4


def test_causal_variants_are_found(reference):
    causal = set(pandas.read_csv(REF_DIR / "causal_vars.csv")["id"])
    res = calc_gwas(
        reference["variants"],
        reference["pheno"]["cont"],
        trait="continuous",
        covariates=reference["covariates"],
        kinship=reference["kinship"],
    )
    top = set(res.stats.nsmallest(10, "p_value")["id"])
    assert len(causal & top) >= 3


def test_grammar_gamma_approx(reference):
    kwargs = dict(
        trait="continuous",
        covariates=reference["covariates"],
        kinship=reference["kinship"],
    )
    exact = calc_gwas(reference["variants"], reference["pheno"]["cont"], **kwargs)
    approx = calc_gwas(
        reference["variants"],
        reference["pheno"]["cont"],
        use_grammar_gamma_approx=True,
        **kwargs,
    )
    assert approx.used_grammar_gamma_approx
    assert not exact.used_grammar_gamma_approx
    assert numpy.allclose(approx.stats["beta"], exact.stats["beta"], rtol=0.5)
    log_ratio = numpy.log10(approx.stats["p_value"] / exact.stats["p_value"])
    assert numpy.abs(numpy.median(log_ratio)) < 0.1
    assert numpy.abs(log_ratio).max() < 1.5

    with pytest.raises(ValueError, match="GRAMMAR-Gamma"):
        calc_gwas(
            reference["variants"],
            reference["pheno"]["cont"],
            trait="continuous",
            use_grammar_gamma_approx=True,
        )


def test_reml_identity(reference):
    # the REML estimate of the genetic variance makes y'Py the residual
    # degrees of freedom
    pheno = reference["pheno"]
    y = pheno["cont"].to_numpy()
    design = numpy.column_stack([numpy.ones(len(y)), pheno[["cov1", "cov2"]]])
    null = _LMMNull(y, design, reference["kinship"].matrix.to_numpy(), TestType.WALD)
    assert abs(null.ypy - (len(y) - 3)) < 1e-6


def test_samples_with_phenotype(reference):
    variants = reference["variants"]
    pheno = reference["pheno"]["cont"].copy()
    kinship = reference["kinship"]
    # some samples have no phenotype, they are left out
    pheno.iloc[::7] = numpy.nan
    pheno = pheno.drop(pheno.index[3:6])
    with_pheno = pheno.dropna().index
    res = calc_gwas(
        variants,
        pheno,
        trait="continuous",
        covariates=reference["covariates"],
        kinship=kinship,
    )
    assert res.samples == tuple(s for s in variants.samples if s in set(with_pheno))
    assert res.null_model.num_samples == len(with_pheno)
    # and that is the same as testing the variants of those samples
    fewer_vars = filter_samples(variants, list(with_pheno))
    res2 = calc_gwas(
        fewer_vars,
        pheno,
        trait="continuous",
        covariates=reference["covariates"],
        kinship=kinship,
    )
    pandas.testing.assert_frame_equal(res.stats, res2.stats)
    # the order of the phenotype does not matter
    res3 = calc_gwas(
        variants,
        pheno.dropna()[::-1],
        trait="continuous",
        covariates=reference["covariates"],
        kinship=kinship,
    )
    pandas.testing.assert_frame_equal(res.stats, res3.stats)

    with pytest.raises(ValueError, match="not in the variants"):
        calc_gwas(
            variants,
            pandas.Series([1.0, 2.0], index=["nobody", "s000"]),
            trait="continuous",
        )
    with pytest.raises(ValueError, match="no covariates"):
        calc_gwas(
            variants,
            pheno,
            trait="continuous",
            covariates=reference["covariates"].iloc[:100],
        )
    with pytest.raises(ValueError, match="0 or 1"):
        calc_gwas(variants, reference["pheno"]["cont"], trait="binomial")
    with pytest.raises(ValueError, match="only has the Wald"):
        calc_gwas(
            variants, reference["pheno"]["cont"], trait="continuous", test="score"
        )
    with pytest.raises(ValueError, match="only have a score"):
        calc_gwas(
            variants,
            reference["pheno"]["binom"],
            trait="binomial",
            kinship=kinship,
            test="wald",
        )
    with pytest.raises(ValueError, match="collinear"):
        covs = reference["covariates"].assign(twice=lambda df: df["cov1"] * 2)
        calc_gwas(
            variants, reference["pheno"]["cont"], trait="continuous", covariates=covs
        )


@pytest.mark.parametrize("trait", ["continuous", "binomial"])
@pytest.mark.parametrize("with_kinship", [False, True])
def test_chunks_and_threads_do_not_matter(reference, trait, with_kinship):
    pheno = reference["pheno"]["cont" if trait == "continuous" else "binom"]
    kinship = reference["kinship"] if with_kinship else None
    kwargs = dict(trait=trait, covariates=reference["covariates"], kinship=kinship)
    res = calc_gwas(reference["variants"], pheno, **kwargs)
    small_chunks = Variants.from_vars(
        reference["variants"], desired_num_vars_per_chunk=77
    )
    res2 = calc_gwas(small_chunks, pheno, num_threads=3, **kwargs)
    pandas.testing.assert_frame_equal(res.stats, res2.stats)
    pandas.testing.assert_series_equal(
        res.null_model.covariate_effects, res2.null_model.covariate_effects
    )
    assert res.null_model.genetic_variance == res2.null_model.genetic_variance


def test_monomorphic_and_missing_variants():
    rng = numpy.random.default_rng(0)
    num_vars, num_samples = 50, 60
    gts = rng.integers(0, 2, size=(num_vars, num_samples, 2)).astype(numpy.int8)
    gts[0, :, :] = 0
    gts[1, :, :] = -1
    gts[2, :5, :] = -1
    samples = create_sample_names(num_samples)
    variants = Variants.from_gt_array(gts, samples=samples)
    pheno = pandas.Series(rng.standard_normal(num_samples), index=samples)
    res = calc_gwas(variants, pheno, trait="continuous")
    assert "chrom" not in res.stats.columns
    assert (
        res.stats["p_value"].isna().to_numpy().tolist() == [True, True] + [False] * 48
    )
    assert res.stats["p_value"][2:].between(0, 1).all()
    kinship = calc_kinship(variants)
    assert kinship.num_vars == 48
    res = calc_gwas(variants, pheno, trait="continuous", kinship=kinship)
    assert res.stats["p_value"][2:].between(0, 1).all()


def test_distributions():
    pytest.importorskip("scipy")
    from scipy import special, stats

    rng = numpy.random.default_rng(0)
    x = rng.uniform(0, 1, 1000)
    for a, b in [(0.5, 0.5), (10, 0.5), (98.5, 0.5), (2.5, 7)]:
        assert numpy.abs(_betainc(a, b, x) - special.betainc(a, b, x)).max() < 1e-12
    t = numpy.concatenate([rng.standard_normal(1000) * 3, [10, 20, 40]])
    for df in [5, 17, 197]:
        ref = 2 * stats.t.sf(numpy.abs(t), df)
        assert numpy.abs(_t_sf_two_sided(t, df) / ref - 1).max() < 1e-10
    chi2 = numpy.concatenate([rng.chisquare(1, 1000), [30, 50, 100]])
    assert numpy.abs(_chi2_sf_1df(chi2) / stats.chi2.sf(chi2, 1) - 1).max() < 1e-12
