import numpy
import pytest

from pynei import (
    Variants,
    calc_jost_dest_pop_dists,
    calc_pairwise_kosman_dists,
    calc_per_sample_stats,
    calc_per_var_distribs,
    do_pca_from_variants,
)
from pynei.config import MAP_REDUCE_CHUNK_SIZE
from pynei.pca import create_012_gt_matrix
from .var_generators import create_sample_names

NUM_SAMPLES = 20
SAMPLES = create_sample_names(NUM_SAMPLES)
POPS = {"pop1": SAMPLES[:10], "pop2": SAMPLES[10:]}


def _create_vars(chunk_size=20):
    rng = numpy.random.default_rng(42)
    gt_array = rng.integers(0, 2, size=(200, NUM_SAMPLES, 2))
    variants = Variants.from_gt_array(gt_array, samples=SAMPLES)
    variants.desired_num_vars_per_chunk = chunk_size
    return variants


def test_one_chunk_per_thread_at_a_time():
    # a variant chunk is already a big unit of work, so handing out several of
    # them to a thread at a time only leaves the other threads with nothing
    assert MAP_REDUCE_CHUNK_SIZE == 1


@pytest.mark.parametrize("num_threads", [2, 4])
def test_per_var_distribs_with_threads(num_threads):
    serial = calc_per_var_distribs(_create_vars(), pops=POPS, min_num_samples=1)
    threaded = calc_per_var_distribs(
        _create_vars(), pops=POPS, min_num_samples=1, num_threads=num_threads
    )
    for stat in ("obs_het", "maf", "exp_het"):
        assert numpy.allclose(getattr(serial, stat).mean, getattr(threaded, stat).mean)
        assert numpy.array_equal(
            getattr(serial, stat).hist_counts.values,
            getattr(threaded, stat).hist_counts.values,
        )
    assert numpy.allclose(
        serial.poly_vars_ratio.poly_ratio, threaded.poly_vars_ratio.poly_ratio
    )


@pytest.mark.parametrize("num_threads", [2, 4])
def test_per_sample_stats_with_threads(num_threads):
    serial = calc_per_sample_stats(_create_vars())
    threaded = calc_per_sample_stats(_create_vars(), num_threads=num_threads)
    assert serial.equals(threaded)


@pytest.mark.parametrize("num_threads", [2, 4])
def test_kosman_dists_with_threads(num_threads):
    serial = calc_pairwise_kosman_dists(_create_vars())
    threaded = calc_pairwise_kosman_dists(_create_vars(), num_threads=num_threads)
    assert numpy.allclose(serial.dist_vector, threaded.dist_vector)


@pytest.mark.parametrize("num_threads", [2, 4])
def test_jost_dest_dists_with_threads(num_threads):
    serial = calc_jost_dest_pop_dists(_create_vars(), pops=POPS, min_num_samples=1)
    threaded = calc_jost_dest_pop_dists(
        _create_vars(), pops=POPS, min_num_samples=1, num_threads=num_threads
    )
    assert numpy.allclose(serial.dist_vector, threaded.dist_vector)


@pytest.mark.parametrize("num_threads", [2, 4])
def test_the_012_matrix_and_the_pca_with_threads(num_threads):
    serial = create_012_gt_matrix(_create_vars())
    threaded = create_012_gt_matrix(_create_vars(), num_threads=num_threads)
    # the rows are the variants, in order, whatever the threads do
    assert numpy.array_equal(serial, threaded)

    assert numpy.allclose(
        do_pca_from_variants(_create_vars()).projections,
        do_pca_from_variants(_create_vars(), num_threads=num_threads).projections,
    )


def test_the_threads_do_not_depend_on_how_the_variants_are_chunked():
    expected = calc_per_var_distribs(_create_vars(200), min_num_samples=1).maf.mean
    for chunk_size in (1, 7, 50):
        got = calc_per_var_distribs(
            _create_vars(chunk_size), min_num_samples=1, num_threads=4
        ).maf.mean
        assert numpy.allclose(got, expected)
