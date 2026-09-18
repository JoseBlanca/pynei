# The R side of make_reference.py, run in its work dir
suppressMessages({library(GMMAT); library(rrBLUP)})

pheno <- read.table("pheno.txt", header = TRUE, comment.char = "")
covar <- read.table("covar.txt", header = TRUE, comment.char = "")
data <- merge(pheno, covar, by = "IID")
data <- data[match(pheno$IID, data$IID), ]
rownames(data) <- data$IID

# the kinship is the one plink2 wrote, so that the mixed models are tested
# with a kinship that does not come from pynei
K <- as.matrix(read.table("sim.rel"))
ids <- read.table("sim.rel.id", header = TRUE, comment.char = "")
rownames(K) <- ids[[ncol(ids)]]
colnames(K) <- ids[[ncol(ids)]]

dosages <- read.csv("dosages.csv", row.names = 1, check.names = FALSE)
dosages <- as.matrix(dosages)[, pheno$IID]

# LMM and GLMM null models and score tests, GMMAT (Chen et al. 2016)
lmm <- glmmkin(cont ~ cov1 + cov2, data = data, kins = K, id = "IID",
               family = gaussian(link = "identity"))
glmm <- glmmkin(binom ~ cov1 + cov2, data = data, kins = K, id = "IID",
                family = binomial(link = "logit"))
glmm.score(lmm, infile = "sim", outfile = "gmmat_lmm_score.tsv", MAF.range = c(0, 1))
glmm.score(glmm, infile = "sim", outfile = "gmmat_glmm_score.tsv", MAF.range = c(0, 1))
# the same tests with some genotypes missing, GMMAT gives a missing genotype
# the mean of its variant (missing.method = "impute2mean")
glmm.score(lmm, infile = "sim_missing", outfile = "gmmat_lmm_score_missing.tsv", MAF.range = c(0, 1))
glmm.score(glmm, infile = "sim_missing", outfile = "gmmat_glmm_score_missing.tsv", MAF.range = c(0, 1))
null <- data.frame(
  model = c("lmm", "glmm"),
  tau = c(lmm$theta[2], glmm$theta[2]),
  sigma2 = c(lmm$theta[1], glmm$theta[1]),
  intercept = c(lmm$coefficients[1], glmm$coefficients[1]),
  cov1 = c(lmm$coefficients[2], glmm$coefficients[2]),
  cov2 = c(lmm$coefficients[3], glmm$coefficients[3])
)
write.table(null, "r_null_models.tsv", sep = "\t", quote = FALSE, row.names = FALSE)

# LMM with the variance components fixed at the null, rrBLUP P3D. rrBLUP
# takes every fixed effect as a factor, so only the binary covariate goes in
geno <- data.frame(marker = rownames(dosages), chrom = 1, pos = seq_len(nrow(dosages)),
                   dosages - 1, check.names = FALSE)
ph <- data.frame(line = data$IID, cont = data$cont, cov2 = data$cov2)
rr <- GWAS(ph, geno, fixed = "cov2", K = K, n.PC = 0, min.MAF = 0, P3D = TRUE, plot = FALSE)
write.table(rr, "rrblup_lmm.tsv", sep = "\t", quote = FALSE, row.names = FALSE)

# the plain logistic regression score test, one glm per variant
score <- t(sapply(seq_len(nrow(dosages)), function(idx) {
  g <- dosages[idx, ]
  if (var(g) == 0) return(c(NA, NA))
  fit <- glm(binom ~ cov1 + cov2 + g, data = cbind(data, g = g), family = binomial)
  an <- anova(fit, test = "Rao")
  c(an["g", "Rao"], an["g", "Pr(>Chi)"])
}))
write.table(data.frame(id = rownames(dosages), score = score[, 1], p = score[, 2]),
            "r_glm_score.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
