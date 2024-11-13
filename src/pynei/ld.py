import numpy

DDOF = 1


def _calc_rogers_huff_r2(gts1, gts2, check_no_mafs_above=0.95, debug=False):
    # print("aquí hay que comprobar que ni gts1 ni gts2 tienen mafs muy grandes")

    covars = numpy.cov(gts1, gts2, ddof=DDOF)
    n_vars1 = gts1.shape[0]
    n_vars2 = gts2.shape[0]
    if debug:
        print("nvars", n_vars1, n_vars2)
    variances = numpy.diag(covars)
    vars1 = variances[:n_vars1]
    vars2 = variances[n_vars1:]
    if debug:
        print("vars1", vars1)
        print("vars2", vars2)

    covars = covars[:n_vars1, n_vars1:]
    if debug:
        print("covars", covars)

    vars1 = numpy.repeat(vars1, n_vars2).reshape((n_vars1, n_vars2))
    vars2 = numpy.tile(vars2, n_vars1).reshape((n_vars1, n_vars2))
    with numpy.errstate(divide="ignore", invalid="ignore"):
        rogers_huff_r = covars / numpy.sqrt(vars1 * vars2)
    # print(vars1)
    # print(vars2)
    return rogers_huff_r
