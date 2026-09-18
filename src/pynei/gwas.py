from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence
import math
import warnings

import numpy
import pandas

from pynei.config import (
    MISSING_ALLELE,
    VAR_TABLE_CHROM_COL,
    VAR_TABLE_POS_COL,
    VAR_TABLE_ID_COL,
)
from pynei.pipeline import Pipeline, run_chunk_calcs
from pynei.pca import _create_pc_names


class TraitType(StrEnum):
    CONTINUOUS = "continuous"
    "A quantitative trait, fitted with a linear model"

    BINOMIAL = "binomial"
    "A 0/1 trait, fitted with a logistic model"


class TestType(StrEnum):
    WALD = "wald"
    "The model is fitted with the variant in it, and its effect is tested"

    SCORE = "score"
    "The model is fitted once without any variant, the score test of Rao"


class GWASModel(StrEnum):
    """The four models, by the names the literature gives them."""

    LM = "lm"
    "Linear model, continuous trait, no kinship"

    LMM = "lmm"
    "Linear mixed model, continuous trait, kinship as a random effect"

    GLM = "glm"
    "Logistic regression, binomial trait, no kinship"

    GLMM = "glmm"
    "Logistic mixed model, binomial trait, kinship as a random effect"


# the number of variants the GRAMMAR-Gamma factor is estimated from
NUM_VARS_FOR_GAMMA = 100
# the REML of the mixed models is a search over log(delta), delta being the
# residual variance over the genetic one
LOG_DELTA_RANGE = (-10.0, 10.0)
GLMM_MAX_ITER = 200
GLMM_TOL = 1e-6
GLM_MAX_ITER = 50
GLM_TOL = 1e-8


@dataclass(frozen=True)
class Kinship:
    """A genomic relationship matrix between the samples.

    It is the matrix of VanRaden and GCTA, that plink2 --make-rel also
    calculates: the standardized dosages of the variants multiplied by
    themselves and divided by the number of variants. An entry is twice the
    coancestry of two samples, and the diagonal is one plus the inbreeding.
    """

    matrix: pandas.DataFrame
    "samples x samples, with the samples as index and columns"

    num_vars: int
    "How many variants it was calculated from"

    @property
    def samples(self) -> tuple:
        return tuple(self.matrix.index)

    def principal_components(self, num_pcs: int) -> pandas.DataFrame:
        """The top principal components of the samples, from the kinship.

        The eigenvectors of the kinship are the principal components of the
        standardized genotypes, so this is the PCA of the variants the
        kinship was calculated from, without going over them again. They are
        given as a DataFrame indexed by sample, so that they can be joined
        to the covariates of calc_gwas.
        """
        eigvals, eigvecs = numpy.linalg.eigh(self.matrix.to_numpy())
        order = numpy.argsort(eigvals)[::-1][:num_pcs]
        projections = eigvecs[:, order] * numpy.sqrt(numpy.abs(eigvals[order]))
        return pandas.DataFrame(
            projections, index=self.matrix.index, columns=_create_pc_names(num_pcs)
        )

    def filter_samples(self, samples: Sequence[str]) -> "Kinship":
        missing = [sample for sample in samples if sample not in self.matrix.index]
        if missing:
            raise ValueError(f"These samples are not in the kinship: {missing}")
        matrix = self.matrix.loc[list(samples), list(samples)]
        return Kinship(matrix=matrix, num_vars=self.num_vars)


@dataclass(frozen=True)
class NullModel:
    """The model fitted once, without any variant in it."""

    model: GWASModel

    covariate_effects: pandas.Series
    "The effect of the intercept and of every covariate"

    residual_variance: float | None
    "The variance not explained by the covariates or the kinship, None for a binomial trait"

    genetic_variance: float | None
    "The variance of the random effect of the kinship, None without kinship"

    heritability: float | None
    "genetic_variance / (genetic_variance + residual_variance), only for the lmm"

    num_samples: int


@dataclass(frozen=True)
class GWASResult:
    """The result of calc_gwas."""

    stats: pandas.DataFrame
    """One row per variant: the chrom, pos and id when the variants have
    them, allele_freq, beta, se and p_value. beta is the effect of one more
    copy of a non major allele, in the units of the trait for a continuous
    one and as a log odds ratio for a binomial one. A variant with no
    variation among the samples tested has nan in beta, se and p_value."""

    null_model: NullModel

    trait: TraitType
    test: TestType
    samples: tuple
    "The samples the trait was tested on, the ones with a phenotype"

    used_grammar_gamma_approx: bool


# The dosages of a chunk


def _calc_dosages(chunk, sample_idxs):
    """The number of non major alleles of every genotype, as floats.

    A missing genotype is given the mean dosage of its variant, so once the
    variant is centered it does not pull the sample in any direction, as the
    PCA does. It gives the dosages, vars x samples, their means, and which
    variants vary at all among these samples.
    """
    gts = chunk.gts
    if sample_idxs is not None:
        gts = gts.filter_samples_with_idxs(sample_idxs)
    gts012 = gts.to_012()
    is_missing = gts012 == MISSING_ALLELE
    dosages = gts012.astype(numpy.float64)
    if is_missing.any():
        dosages[is_missing] = numpy.nan
        with warnings.catch_warnings():
            # a variant with only missing genotypes has no mean
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = numpy.nanmean(dosages, axis=1)
        all_missing = numpy.isnan(means)
        means[all_missing] = 0.0
        dosages = numpy.where(is_missing, means[:, None], dosages)
    else:
        means = dosages.mean(axis=1)
    is_poly = dosages.std(axis=1) > 0
    return dosages, means, is_poly


# The kinship


class _KinshipCalc:
    def __init__(self, sample_idxs, samples, ploidy):
        self.sample_idxs = sample_idxs
        self.samples = samples
        self.ploidy = ploidy

    def calc_for_chunk(self, chunk, cache):
        gts = chunk.gts
        if self.sample_idxs is not None:
            gts = gts.filter_samples_with_idxs(self.sample_idxs)
        dosages, means, is_poly = _calc_dosages(chunk, self.sample_idxs)
        dosages = dosages[is_poly]
        means = means[is_poly]
        freqs = means / self.ploidy
        stdevs = numpy.sqrt(self.ploidy * freqs * (1 - freqs))
        standardized = (dosages - means[:, None]) / stdevs[:, None]
        num_vars = int(is_poly.sum())
        # a missing genotype adds nothing to a pair, it is the mean, and it
        # is not counted for that pair either, as GCTA and plink2 do it
        is_called = ~numpy.any(gts.missing_mask, axis=2)[is_poly]
        if is_called.all():
            num_vars_per_pair = num_vars
        else:
            is_called = is_called.astype(numpy.float64)
            num_vars_per_pair = is_called.T @ is_called
        return {
            "zz": standardized.T @ standardized,
            "num_vars": num_vars,
            "num_vars_per_pair": num_vars_per_pair,
        }

    def reduce(self, accumulated, contribution):
        return {
            "zz": accumulated["zz"] + contribution["zz"],
            "num_vars": accumulated["num_vars"] + contribution["num_vars"],
            "num_vars_per_pair": accumulated["num_vars_per_pair"]
            + contribution["num_vars_per_pair"],
        }

    def finish(self, accumulated):
        num_vars = accumulated["num_vars"]
        if not num_vars:
            raise ValueError("No variant varies among the samples, there is no kinship")
        with numpy.errstate(invalid="ignore", divide="ignore"):
            matrix = accumulated["zz"] / accumulated["num_vars_per_pair"]
        matrix = pandas.DataFrame(matrix, index=self.samples, columns=self.samples)
        return Kinship(matrix=matrix, num_vars=num_vars)


def calc_kinship(
    variants, samples: Sequence[str] | None = None, num_threads: int = 1
) -> Kinship:
    """It calculates the genomic relationship matrix of the samples.

    Every variant that varies is centered and standardized by the variance
    its allele frequency gives it, and the kinship is the standardized
    dosages multiplied by themselves and divided by the number of variants,
    as GCTA and plink2 --make-rel calculate it. A missing genotype adds
    nothing to its pairs and is not counted in their number of variants, so
    every pair is divided by the variants called in both of its samples. One
    pass over the variants, and a samples x samples matrix that is
    accumulated chunk by chunk.

    It is normally calculated from variants pruned by linkage disequilibrium,
    filter_by_ld_and_maf, and then given to calc_gwas to test all of them.
    """
    all_samples = variants.samples
    if samples is None:
        sample_idxs = None
        samples = all_samples
    else:
        samples = tuple(samples)
        sample_idxs = _get_sample_idxs(samples, all_samples)
        if sample_idxs == list(range(len(all_samples))):
            sample_idxs = None
    calc = _KinshipCalc(sample_idxs, samples, variants.ploidy)
    return run_chunk_calcs(variants, {"kinship": calc}, num_threads=num_threads)[
        "kinship"
    ]


# Distributions. numpy has no special functions, so the two that the p-values
# need are here: erfc comes from math, and the regularized incomplete beta is
# the continued fraction of Numerical Recipes


_erfc = numpy.frompyfunc(math.erfc, 1, 1)


def _chi2_sf_1df(chi2):
    """P(X > chi2) for a chi2 with one degree of freedom."""
    chi2 = numpy.asarray(chi2, dtype=numpy.float64)
    return _erfc(numpy.sqrt(chi2 / 2)).astype(numpy.float64)


def _betainc_continued_fraction(a, b, x, max_iter=500, eps=1e-15):
    """The continued fraction of the incomplete beta, Lentz's method, over an array x."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = numpy.ones_like(x)
    d = 1 - qab * x / qap
    d = numpy.where(numpy.abs(d) < tiny, tiny, d)
    d = 1 / d
    h = d.copy()
    not_converged = numpy.ones(x.shape, dtype=bool)
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = numpy.where(numpy.abs(d) < tiny, tiny, d)
        c = 1 + aa / c
        c = numpy.where(numpy.abs(c) < tiny, tiny, c)
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = numpy.where(numpy.abs(d) < tiny, tiny, d)
        c = 1 + aa / c
        c = numpy.where(numpy.abs(c) < tiny, tiny, c)
        d = 1 / d
        delta = d * c
        h *= delta
        not_converged &= numpy.abs(delta - 1) >= eps
        if not not_converged.any():
            break
    return h


def _betainc(a: float, b: float, x):
    """The regularized incomplete beta function I_x(a, b), for an array x."""
    x = numpy.asarray(x, dtype=numpy.float64)
    result = numpy.empty_like(x)
    result[x <= 0] = 0.0
    result[x >= 1] = 1.0
    inside = (x > 0) & (x < 1)
    if not inside.any():
        return result
    xi = x[inside]
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * numpy.log(xi)
        + b * numpy.log1p(-xi)
    )
    front = numpy.exp(log_front)
    # the continued fraction converges fast for x below (a + 1) / (a + b + 2)
    # and the symmetry of the function covers the rest
    direct = xi < (a + 1) / (a + b + 2)
    res = numpy.empty_like(xi)
    if direct.any():
        res[direct] = front[direct] * _betainc_continued_fraction(a, b, xi[direct]) / a
    if (~direct).any():
        res[~direct] = (
            1 - front[~direct] * _betainc_continued_fraction(b, a, 1 - xi[~direct]) / b
        )
    result[inside] = res
    return result


def _t_sf_two_sided(t, df: int):
    """P(|T| > t) for a Student t with df degrees of freedom."""
    t = numpy.asarray(t, dtype=numpy.float64)
    return _betainc(df / 2, 0.5, df / (df + t * t))


# The null models


def _add_intercept(covariates: numpy.ndarray) -> numpy.ndarray:
    num_samples = covariates.shape[0]
    return numpy.column_stack([numpy.ones(num_samples), covariates])


def _stats_frame(beta, se, p_value):
    return {"beta": beta, "se": se, "p_value": p_value}


def _nan_stats(num_vars):
    nans = numpy.full(num_vars, numpy.nan)
    return _stats_frame(nans, nans.copy(), nans.copy())


class _LMNull:
    """The linear model without kinship, the Wald test is the t test of lm."""

    model = GWASModel.LM

    def __init__(self, y, design):
        self.num_samples, num_coefs = design.shape
        q, r = numpy.linalg.qr(design)
        self.q = q
        self.coefs = numpy.linalg.solve(r, q.T @ y)
        self.resid = y - q @ (q.T @ y)
        self.rss = float(self.resid @ self.resid)
        self.df_null = self.num_samples - num_coefs
        self.df = self.df_null - 1
        self.residual_variance = self.rss / self.df_null
        self.genetic_variance = None
        self.heritability = None

    def test_chunk(self, dosages, is_poly):
        # the covariates are taken out of every variant, and then the effect
        # of the variant is the plain slope on the residuals of the trait
        x = dosages[is_poly]
        x = x - (x @ self.q) @ self.q.T
        xx = numpy.einsum("ij,ij->i", x, x)
        num = x @ self.resid
        beta = num / xx
        rss = self.rss - beta * num
        se = numpy.sqrt(rss / self.df / xx)
        p_value = _t_sf_two_sided(beta / se, self.df)
        return beta, se, p_value


def _reml_delta(uy, udesign, eigvals, num_samples, num_coefs):
    """The delta, residual over genetic variance, that maximizes the REML.

    The trait and the design are already rotated by the eigenvectors of the
    kinship, so every evaluation is linear in the samples. EMMA, Kang 2008.
    """

    def neg_reml(log_delta):
        delta = math.exp(log_delta)
        weights = 1 / (eigvals + delta)
        dvd = udesign.T @ (weights[:, None] * udesign)
        coefs = numpy.linalg.solve(dvd, udesign.T @ (weights * uy))
        resid = uy - udesign @ coefs
        quad = float(weights @ (resid * resid))
        return (
            numpy.log(eigvals + delta).sum()
            + (num_samples - num_coefs) * math.log(quad)
            + numpy.linalg.slogdet(dvd)[1]
        )

    # a grid finds the basin and a golden section search settles in it
    grid = numpy.linspace(*LOG_DELTA_RANGE, 101)
    values = [neg_reml(log_delta) for log_delta in grid]
    best = int(numpy.argmin(values))
    low = grid[max(best - 1, 0)]
    high = grid[min(best + 1, len(grid) - 1)]
    golden = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        mid1 = high - golden * (high - low)
        mid2 = low + golden * (high - low)
        if neg_reml(mid1) < neg_reml(mid2):
            high = mid2
        else:
            low = mid1
    return math.exp((low + high) / 2)


class _ProjectionNull:
    """A null model whose tests come from one projection matrix P.

    For the mixed models the statistic of every variant x is built from
    x'Py and x'Px, with P = V^-1 - V^-1 C (C' V^-1 C)^-1 C' V^-1, V being
    the covariance of the trait under the null and C the design. x'Px costs
    a samples x samples product per variant, and the GRAMMAR-Gamma
    approximation replaces it by gamma times x'x, one gamma estimated from
    a few variants.
    """

    def __init__(self):
        self.gamma = None

    def _num_and_den(self, x):
        num = x @ self.py
        if self.gamma is None:
            den = numpy.einsum("ij,ij->i", x @ self.projection, x)
        else:
            centered = x - x.mean(axis=1)[:, None]
            den = self.gamma * numpy.einsum("ij,ij->i", centered, centered)
        return num, den

    def _score_test(self, x):
        # the variance components stay at the null, a chi2 with one degree
        # of freedom, as GMMAT
        num, den = self._num_and_den(x)
        beta = num / den
        se = 1 / numpy.sqrt(den)
        p_value = _chi2_sf_1df(num * num / den)
        return beta, se, p_value

    def estimate_gamma(self, dosages, is_poly):
        x = dosages[is_poly][:NUM_VARS_FOR_GAMMA]
        if not x.shape[0]:
            raise ValueError("No variant varies, the gamma can not be estimated")
        exact = numpy.einsum("ij,ij->i", x @ self.projection, x)
        centered = x - x.mean(axis=1)[:, None]
        approx = numpy.einsum("ij,ij->i", centered, centered)
        self.gamma = float(numpy.mean(exact / approx))


def _projection(vinv, design):
    vinv_design = vinv @ design
    dvd = design.T @ vinv_design
    return vinv - vinv_design @ numpy.linalg.solve(dvd, vinv_design.T)


class _LMMNull(_ProjectionNull):
    """The linear mixed model, REML on the eigendecomposition of the kinship.

    The Wald test keeps the ratio of the two variances at the null and
    estimates the residual variance again with the variant in the model, a
    t test with the degrees of freedom of the fixed effects, as rrBLUP,
    EMMAX and GEMMA do. The score test keeps both variances at the null, a
    chi2, as GMMAT does.
    """

    model = GWASModel.LMM

    def __init__(self, y, design, kinship, test: TestType):
        super().__init__()
        self.test = test
        self.num_samples, num_coefs = design.shape
        self.df = self.num_samples - num_coefs - 1
        eigvals, eigvecs = numpy.linalg.eigh(kinship)
        # a kinship is positive semidefinite, the tiny negative eigenvalues
        # are rounding
        eigvals = numpy.maximum(eigvals, 0.0)
        uy = eigvecs.T @ y
        udesign = eigvecs.T @ design
        delta = _reml_delta(uy, udesign, eigvals, self.num_samples, num_coefs)
        weights = 1 / (eigvals + delta)
        dvd = udesign.T @ (weights[:, None] * udesign)
        self.coefs = numpy.linalg.solve(dvd, udesign.T @ (weights * uy))
        resid = uy - udesign @ self.coefs
        self.genetic_variance = float(weights @ (resid * resid)) / (
            self.num_samples - num_coefs
        )
        self.residual_variance = delta * self.genetic_variance
        self.heritability = self.genetic_variance / (
            self.genetic_variance + self.residual_variance
        )
        vinv = (
            eigvecs / (self.genetic_variance * eigvals + self.residual_variance)
        ) @ eigvecs.T
        self.projection = _projection(vinv, design)
        self.py = self.projection @ y
        # y'Py is the generalized residual sum of squares of the null over
        # the genetic variance, and the REML makes it num_samples - num_coefs
        self.ypy = float(y @ self.py)

    def test_chunk(self, dosages, is_poly):
        x = dosages[is_poly]
        if self.test == TestType.SCORE:
            return self._score_test(x)
        num, den = self._num_and_den(x)
        beta = num / den
        # what the variant leaves of y'Py, over the degrees of freedom, is
        # the residual variance with the variant in, over the genetic one
        se = numpy.sqrt((self.ypy - num * num / den) / (self.df * den))
        p_value = _t_sf_two_sided(beta / se, self.df)
        return beta, se, p_value


def _expit(eta):
    return 1 / (1 + numpy.exp(-eta))


def _fit_logistic(y, design, max_iter=GLM_MAX_ITER, tol=GLM_TOL):
    """Logistic regression by iteratively reweighted least squares."""
    coefs = numpy.zeros(design.shape[1])
    coefs[0] = math.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))
    for _ in range(max_iter):
        mu = _expit(design @ coefs)
        weights = mu * (1 - mu)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (y - mu)
        step = numpy.linalg.solve(hessian, gradient)
        coefs = coefs + step
        if numpy.abs(step).max() < tol:
            break
    else:
        raise RuntimeError("The logistic regression of the null model did not converge")
    mu = _expit(design @ coefs)
    return coefs, mu


class _GLMNull:
    """Logistic regression without kinship.

    The Wald test fits one logistic regression per variant, every variant of
    the chunk at once, starting from the null fit. The score test only needs
    the null, the covariates take the place of the projection: x'Px is
    x'Wx - x'WC (C'WC)^-1 C'Wx, which costs nothing per variant.
    """

    model = GWASModel.GLM

    def __init__(self, y, design, test: TestType):
        self.y = y
        self.design = design
        self.test = test
        self.num_samples, self.num_coefs = design.shape
        self.coefs, self.mu = _fit_logistic(y, design)
        self.weights = self.mu * (1 - self.mu)
        self.resid = y - self.mu
        self.wdesign = self.weights[:, None] * design
        self.dwd_inv = numpy.linalg.inv(design.T @ self.wdesign)
        self.residual_variance = None
        self.genetic_variance = None
        self.heritability = None

    def test_chunk(self, dosages, is_poly):
        x = dosages[is_poly]
        if self.test == TestType.SCORE:
            return self._score_test(x)
        return self._wald_test(x)

    def _score_test(self, x):
        num = x @ self.resid
        xwd = x @ self.wdesign
        den = numpy.einsum("ij,ij->i", x * self.weights, x) - numpy.einsum(
            "ij,jk,ik->i", xwd, self.dwd_inv, xwd
        )
        beta = num / den
        se = 1 / numpy.sqrt(den)
        return beta, se, _chi2_sf_1df(num * num / den)

    def _wald_test(self, x, max_iter=GLM_MAX_ITER, tol=GLM_TOL):
        num_vars = x.shape[0]
        design = self.design
        num_coefs = self.num_coefs
        # one coefficient vector per variant: the covariates and the variant
        coefs = numpy.tile(numpy.append(self.coefs, 0.0), (num_vars, 1))
        eta_design = design @ coefs[:, :num_coefs].T  # samples x vars
        # the outer products of the design rows, to weight them per variant
        design_outer = (design[:, :, None] * design[:, None, :]).reshape(
            self.num_samples, num_coefs * num_coefs
        )
        active = numpy.ones(num_vars, dtype=bool)
        diverged = numpy.zeros(num_vars, dtype=bool)
        hessian = numpy.empty((num_vars, num_coefs + 1, num_coefs + 1))
        for _ in range(max_iter):
            eta = eta_design.T + coefs[:, -1][:, None] * x  # vars x samples
            mu = _expit(eta)
            weights = mu * (1 - mu)
            resid = self.y[None, :] - mu
            hessian[:, :num_coefs, :num_coefs] = (weights @ design_outer).reshape(
                num_vars, num_coefs, num_coefs
            )
            xw = x * weights
            xwd = xw @ design
            hessian[:, :num_coefs, -1] = xwd
            hessian[:, -1, :num_coefs] = xwd
            hessian[:, -1, -1] = numpy.einsum("ij,ij->i", xw, x)
            gradient = numpy.empty((num_vars, num_coefs + 1))
            gradient[:, :num_coefs] = resid @ design
            gradient[:, -1] = numpy.einsum("ij,ij->i", x, resid)
            idxs = numpy.flatnonzero(active)
            try:
                step = numpy.linalg.solve(hessian[idxs], gradient[idxs][:, :, None])[
                    :, :, 0
                ]
            except numpy.linalg.LinAlgError:
                step = numpy.full((idxs.size, num_coefs + 1), numpy.nan)
                for pos, idx in enumerate(idxs):
                    try:
                        step[pos] = numpy.linalg.solve(hessian[idx], gradient[idx])
                    except numpy.linalg.LinAlgError:
                        pass
            bad = ~numpy.isfinite(step).all(axis=1)
            step[bad] = 0.0
            diverged[idxs[bad]] = True
            coefs[idxs] += step
            eta_design[:, idxs] = design @ coefs[idxs, :num_coefs].T
            converged = numpy.abs(step).max(axis=1) < tol
            active[idxs[converged | bad]] = False
            # a variant that separates the cases from the controls has no
            # finite effect, its coefficient runs away
            runaway = numpy.abs(coefs[:, -1]) > 30
            diverged |= runaway
            active &= ~runaway
            if not active.any():
                break
        diverged |= active
        beta = coefs[:, -1]
        with numpy.errstate(invalid="ignore", divide="ignore"):
            variances = numpy.linalg.inv(hessian)[:, -1, -1]
        se = numpy.sqrt(variances)
        p_value = _chi2_sf_1df((beta / se) ** 2)
        for values in (beta, se, p_value):
            values[diverged] = numpy.nan
        return beta, se, p_value


class _GLMMNull(_ProjectionNull):
    """The logistic mixed model, penalized quasi-likelihood.

    Breslow and Clayton 1993, as GMMAT fits it: every iteration is a linear
    mixed model on the working trait with the weights of the logistic
    model, and the variance of the kinship effect takes one average
    information REML step. At convergence Py is the residual of the trait,
    y - mu, which is what the score test uses.
    """

    model = GWASModel.GLMM

    def __init__(self, y, design, kinship, max_iter=GLMM_MAX_ITER, tol=GLMM_TOL):
        super().__init__()
        self.num_samples, num_coefs = design.shape
        identity = numpy.eye(self.num_samples)
        coefs, mu = _fit_logistic(y, design)
        eta = design @ coefs
        # the working trait with no kinship effect, and its variance as the
        # first guess of the kinship variance, as GMMAT starts
        weights = mu * (1 - mu)
        working = eta + (y - mu) / weights
        tau = float(numpy.var(working)) / 2
        for _ in range(max_iter):
            weights = mu * (1 - mu)
            working = eta + (y - mu) / weights
            sigma = numpy.diag(1 / weights) + tau * kinship
            sigma_inv = numpy.linalg.inv(sigma)
            projection = _projection(sigma_inv, design)
            sigma_inv_design = sigma_inv @ design
            new_coefs = numpy.linalg.solve(
                design.T @ sigma_inv_design, sigma_inv_design.T @ working
            )
            pw = projection @ working
            kpw = kinship @ pw
            score = 0.5 * (pw @ kpw - numpy.einsum("ij,ji->", projection, kinship))
            ai = 0.5 * (kpw @ (projection @ kpw))
            new_tau = tau + score / ai
            if new_tau < 0:
                new_tau = 0.0
            eta = design @ new_coefs + new_tau * kpw
            mu = _expit(eta)
            change = max(
                numpy.abs(new_coefs - coefs).max() / (numpy.abs(coefs).max() + tol),
                abs(new_tau - tau) / (abs(tau) + tol),
            )
            coefs, tau = new_coefs, new_tau
            if change < tol:
                break
        else:
            raise RuntimeError("The logistic mixed model did not converge")
        self.coefs = coefs
        self.genetic_variance = tau
        self.residual_variance = None
        self.heritability = None
        weights = mu * (1 - mu)
        sigma = numpy.diag(1 / weights) + tau * kinship
        del identity
        self.projection = _projection(numpy.linalg.inv(sigma), design)
        self.py = y - mu

    def test_chunk(self, dosages, is_poly):
        return self._score_test(dosages[is_poly])


# The samples, the trait and the covariates


def _get_sample_idxs(samples, all_samples):
    idx_of_sample = {sample: idx for idx, sample in enumerate(all_samples)}
    missing = [sample for sample in samples if sample not in idx_of_sample]
    if missing:
        raise ValueError(f"These samples are not in the variants: {missing}")
    return [idx_of_sample[sample] for sample in samples]


def _prepare_samples_and_design(variants, phenotype, covariates, trait):
    if isinstance(phenotype, pandas.DataFrame):
        if phenotype.shape[1] != 1:
            raise ValueError(
                "The phenotype should be one Series, or a DataFrame with one column"
            )
        phenotype = phenotype.iloc[:, 0]
    if not isinstance(phenotype, pandas.Series):
        raise ValueError("The phenotype should be a pandas Series indexed by sample")
    if phenotype.index.has_duplicates:
        raise ValueError("There are samples repeated in the phenotype")
    phenotype = phenotype.dropna()
    if not len(phenotype):
        raise ValueError("No sample has a phenotype")
    all_samples = variants.samples
    _get_sample_idxs(phenotype.index, all_samples)
    # the samples are kept in the order the variants have them
    with_phenotype = set(phenotype.index)
    samples = tuple(sample for sample in all_samples if sample in with_phenotype)
    sample_idxs = [
        idx for idx, sample in enumerate(all_samples) if sample in with_phenotype
    ]
    if len(samples) == len(all_samples):
        sample_idxs = None
    y = phenotype.loc[list(samples)].to_numpy(dtype=numpy.float64)
    if trait == TraitType.BINOMIAL:
        if not numpy.isin(y, [0.0, 1.0]).all():
            raise ValueError("A binomial phenotype should be 0 or 1, or False or True")
        if y.min() == y.max():
            raise ValueError("Every sample has the same phenotype")

    if covariates is None:
        design = numpy.ones((len(samples), 1))
        names = ["intercept"]
    else:
        if isinstance(covariates, pandas.Series):
            covariates = covariates.to_frame()
        missing = [sample for sample in samples if sample not in covariates.index]
        if missing:
            raise ValueError(f"These samples have no covariates: {missing}")
        covariates = covariates.loc[list(samples)]
        if covariates.isna().any().any():
            raise ValueError("There are missing values in the covariates")
        try:
            values = covariates.to_numpy(dtype=numpy.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "The covariates should be numeric, code the categorical ones, with pandas.get_dummies for instance"
            ) from error
        design = _add_intercept(values)
        names = ["intercept"] + [str(name) for name in covariates.columns]
    if numpy.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The covariates are collinear, or one of them is constant")
    if design.shape[0] <= design.shape[1] + 1:
        raise ValueError("There are fewer samples than coefficients to fit")
    return samples, sample_idxs, y, design, names


def _fit_null(y, design, names, kinship, trait, test):
    if trait == TraitType.CONTINUOUS:
        if kinship is None:
            null = _LMNull(y, design)
        else:
            null = _LMMNull(y, design, kinship, test)
    else:
        if kinship is None:
            null = _GLMNull(y, design, test)
        else:
            null = _GLMMNull(y, design, kinship)
    null_model = NullModel(
        model=null.model,
        covariate_effects=pandas.Series(null.coefs, index=names),
        residual_variance=null.residual_variance,
        genetic_variance=null.genetic_variance,
        heritability=null.heritability,
        num_samples=null.num_samples,
    )
    return null, null_model


def _default_test(trait, kinship):
    if trait == TraitType.CONTINUOUS or kinship is None:
        return TestType.WALD
    return TestType.SCORE


def _check_test(test, trait, kinship):
    if test is None:
        return _default_test(trait, kinship)
    test = TestType(test)
    if trait == TraitType.CONTINUOUS and kinship is None and test == TestType.SCORE:
        raise ValueError(
            "A continuous trait without kinship only has the Wald test, the t test of the linear model"
        )
    if trait == TraitType.BINOMIAL and kinship is not None and test == TestType.WALD:
        raise ValueError(
            "A binomial trait with a kinship can only have a score test, a Wald test would fit one mixed model per variant"
        )
    return test


def _vars_info_columns(chunk):
    columns = {}
    if chunk.vars_info is not None:
        for col in (VAR_TABLE_CHROM_COL, VAR_TABLE_POS_COL, VAR_TABLE_ID_COL):
            if col in chunk.vars_info.columns:
                columns[col] = chunk.vars_info[col].to_numpy()
    return columns


class _ChunkTester:
    def __init__(self, null, sample_idxs, ploidy):
        self.null = null
        self.sample_idxs = sample_idxs
        self.ploidy = ploidy

    def __call__(self, chunk):
        dosages, means, is_poly = _calc_dosages(chunk, self.sample_idxs)
        num_vars = dosages.shape[0]
        stats = _vars_info_columns(chunk)
        stats["allele_freq"] = means / self.ploidy
        nan_stats = _nan_stats(num_vars)
        if is_poly.any():
            beta, se, p_value = self.null.test_chunk(dosages, is_poly)
            nan_stats["beta"][is_poly] = beta
            nan_stats["se"][is_poly] = se
            nan_stats["p_value"][is_poly] = p_value
        stats.update(nan_stats)
        return pandas.DataFrame(stats)


def calc_gwas(
    variants,
    phenotype: pandas.Series,
    trait: TraitType | str,
    covariates: pandas.DataFrame | None = None,
    kinship: Kinship | None = None,
    test: TestType | str | None = None,
    use_grammar_gamma_approx: bool = False,
    num_threads: int = 1,
) -> GWASResult:
    """It tests the association of every variant with a trait.

    The phenotype is a Series indexed by sample name. The samples without a
    phenotype, missing from it or nan, are left out. A continuous trait is
    fitted with a linear model and a binomial one, 0 or 1, with a logistic
    one. The covariates are a DataFrame indexed by sample, numeric, an
    intercept is always added.

    The population structure is accounted for in one of three ways, and
    which one depends on the samples. With a kinship, calc_kinship, the
    model is a mixed one with the kinship as the covariance of a random
    polygenic effect, which is what structured or related panels use. Without
    it, the top principal components, Kinship.principal_components or
    do_pca_from_variants, can be given as covariates, which is enough for
    unrelated samples. Both at once is the Q+K model for strongly subdivided
    populations.

    The null model, covariates and kinship, is fitted once, and then every
    variant is tested in one pass. With a kinship the ratio of the variance
    components is kept at the null for every variant, P3D, as EMMAX, rrBLUP
    and GMMAT do. The test is Wald where a per variant fit is cheap, a
    continuous trait, or a binomial one without kinship, and the score test
    with a binomial trait and a kinship. A continuous trait with a kinship
    and a binomial one without it can ask for either: the Wald test of the
    linear mixed model estimates the residual variance again for every
    variant, a t test, as rrBLUP does, and its score test keeps it at the
    null, a chi2, as GMMAT does. use_grammar_gamma_approx makes the mixed
    models linear in the samples per variant instead of quadratic, at the
    cost of accuracy when the population is strongly structured.
    """
    trait = TraitType(trait)
    test = _check_test(test, trait, kinship)
    samples, sample_idxs, y, design, names = _prepare_samples_and_design(
        variants, phenotype, covariates, trait
    )
    if kinship is not None:
        kinship_matrix = kinship.filter_samples(samples).matrix.to_numpy(
            dtype=numpy.float64
        )
    else:
        kinship_matrix = None
    null, null_model = _fit_null(y, design, names, kinship_matrix, trait, test)

    if use_grammar_gamma_approx:
        if kinship is None:
            raise ValueError(
                "The GRAMMAR-Gamma approximation is only for the mixed models"
            )
        first_chunk = next(iter(variants.iter_vars_chunks()))
        dosages, _, is_poly = _calc_dosages(first_chunk, sample_idxs)
        null.estimate_gamma(dosages, is_poly)

    pipeline = Pipeline(map_functs=[_ChunkTester(null, sample_idxs, variants.ploidy)])
    chunk_stats = list(pipeline.map_chunks(variants, num_threads=num_threads))
    if not chunk_stats:
        raise ValueError("There are no variants to test")
    stats = pandas.concat(chunk_stats, ignore_index=True)
    return GWASResult(
        stats=stats,
        null_model=null_model,
        trait=trait,
        test=test,
        samples=samples,
        used_grammar_gamma_approx=use_grammar_gamma_approx,
    )
