import numpy

DDOF = 1


def _calc_maf_from_012_gts(gts: numpy.array):
    counts = {}
    for gt in [0, 1, 2]:
        counts[gt] = numpy.sum(gts == gt, axis=1)

    num_gts_per_var = counts[0] + counts[1] + counts[2]
    counts_major_allele = numpy.maximum(counts[0], counts[2])
    freqs_major_allele = counts_major_allele / num_gts_per_var
    freqs_het = counts[1] / num_gts_per_var
    maf_per_var = freqs_major_allele + 0.5 * freqs_het
    return maf_per_var


def _center_gts_for_r(gts_012: numpy.ndarray):
    """The genotypes centred, and the sum of the squares of every row.

    r between two variants needs nothing else of them, and neither of these
    changes while a chunk is walked through, so they are worked out once for
    the whole chunk instead of again on every comparison.
    """
    centered = gts_012.astype(float)
    centered -= centered.mean(axis=1, keepdims=True)
    sum_of_squares = numpy.einsum("ij,ij->i", centered, centered)
    return centered, sum_of_squares


def _calc_r_against_ref(centered, sum_of_squares, ref_centered, ref_sum_of_squares):
    """r between every given variant and the reference one, already centred."""
    covars = centered @ ref_centered
    with numpy.errstate(divide="ignore", invalid="ignore"):
        return covars / numpy.sqrt(sum_of_squares * ref_sum_of_squares)


def _calc_rogers_huff_r2(
    gts1: numpy.ndarray,
    gts2: numpy.ndarray,
    check_no_mafs_above: float | None = 0.95,
    debug=False,
):
    if check_no_mafs_above is not None:
        maf_per_var = _calc_maf_from_012_gts(gts1)
        if numpy.any(maf_per_var > check_no_mafs_above):
            raise ValueError(
                f"There are variations with mafs above {check_no_mafs_above}, filter them out or modify this check"
            )

    # Only the covariance of every variant of gts1 against every variant of
    # gts2 is wanted. numpy.cov(gts1, gts2) stacks them and gives back the
    # whole (n1 + n2) square, so it also works out every pair within gts1 and
    # every pair within gts2 and then they are thrown away. For the one
    # against many that the LD filter asks for, one variant against the 5000
    # of a chunk, that is 25 million covariances computed and 5000 used.
    #
    # r does not need the covariances themselves anyway. With the variants
    # centred, r = sum(x * y) / sqrt(sum(x * x) * sum(y * y)), and the 1 / (n
    # - ddof) of every covariance cancels out between the top and the bottom.
    gts1, vars1 = _center_gts_for_r(gts1)
    gts2, vars2 = _center_gts_for_r(gts2)

    covars = gts1 @ gts2.T
    if debug:
        print("nvars", gts1.shape[0], gts2.shape[0])
        print("vars1", vars1 / (gts1.shape[1] - DDOF))
        print("vars2", vars2 / (gts2.shape[1] - DDOF))
        print("covars", covars / (gts1.shape[1] - DDOF))

    with numpy.errstate(divide="ignore", invalid="ignore"):
        rogers_huff_r = covars / numpy.sqrt(vars1[:, None] * vars2[None, :])
    return rogers_huff_r
