import tempfile
from pathlib import Path

import numpy
import pytest

from pynei import Variants, load_vars, vars_from_vcf, write_vars
from pynei.config import (
    DEF_NUM_GTS_PER_CHUNK,
    MAX_NUM_VARS_PER_CHUNK,
    MIN_NUM_VARS_PER_CHUNK,
)
from pynei.var_filters import filter_by_missing_data, filter_samples
from pynei.variants import calc_num_vars_per_chunk
from .test_vcf import VCF_45
from .var_generators import create_sample_names


def test_the_chunks_are_sized_by_their_genotypes():
    # what a chunk costs, in memory and in work, is its genotypes, variants
    # times samples, so that is what is kept about the same
    for num_samples in (500, 1_000, 10_000, 50_000):
        num_vars = calc_num_vars_per_chunk(num_samples)
        assert num_vars * num_samples == pytest.approx(DEF_NUM_GTS_PER_CHUNK, rel=0.01)


def test_the_number_of_vars_per_chunk_is_capped():
    # with few samples the budget would ask for so many variants that a normal
    # dataset would be a couple of chunks, and the threads would have nothing
    # to share
    for num_samples in (1, 10, 100):
        assert calc_num_vars_per_chunk(num_samples) == MAX_NUM_VARS_PER_CHUNK


def test_the_number_of_vars_per_chunk_has_a_minimum():
    # with very many samples the budget would ask for so few variants that
    # every chunk would carry its own overhead for almost nothing
    for num_samples in (500_000, DEF_NUM_GTS_PER_CHUNK * 10):
        assert calc_num_vars_per_chunk(num_samples) == MIN_NUM_VARS_PER_CHUNK


def _create_vars(num_samples, num_vars=50):
    gt_array = numpy.random.randint(0, 2, size=(num_vars, num_samples, 2))
    return Variants.from_gt_array(gt_array, samples=create_sample_names(num_samples))


def test_the_variants_size_their_chunks_from_their_samples():
    for num_samples in (100, 1_000, 10_000):
        variants = _create_vars(num_samples, num_vars=2)
        assert variants.desired_num_vars_per_chunk == calc_num_vars_per_chunk(
            num_samples
        )


def test_the_chunk_size_can_be_set():
    variants = _create_vars(1_000)
    assert variants.desired_num_vars_per_chunk == calc_num_vars_per_chunk(1_000)

    variants.desired_num_vars_per_chunk = 7
    assert variants.desired_num_vars_per_chunk == 7
    assert [chunk.num_vars for chunk in variants.iter_vars_chunks()][0] == 7

    # setting it back to None gives the worked out one again
    variants.desired_num_vars_per_chunk = None
    assert variants.desired_num_vars_per_chunk == calc_num_vars_per_chunk(1_000)


def test_filtering_the_samples_sizes_the_chunks_again():
    variants = _create_vars(10_000)
    assert variants.desired_num_vars_per_chunk == calc_num_vars_per_chunk(10_000)

    # fewer samples means the same genotypes fit in more variants
    fewer = filter_samples(variants, samples=variants.samples[:1_000])
    assert fewer.desired_num_vars_per_chunk == calc_num_vars_per_chunk(1_000)
    assert fewer.desired_num_vars_per_chunk > variants.desired_num_vars_per_chunk


def test_a_chunk_size_that_was_set_survives_the_filters():
    variants = _create_vars(10_000)
    variants.desired_num_vars_per_chunk = 3
    filtered = filter_by_missing_data(variants, max_allowed_missing_rate=1)
    assert filtered.desired_num_vars_per_chunk == 3
    assert (
        filter_samples(filtered, variants.samples[:10]).desired_num_vars_per_chunk == 3
    )


def test_the_sources_size_their_chunks_too():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        vcf_path = tempdir / "variants.vcf"
        vcf_path.write_bytes(VCF_45)
        from_vcf = vars_from_vcf(vcf_path)
        num_samples = from_vcf.num_samples
        assert from_vcf.desired_num_vars_per_chunk == calc_num_vars_per_chunk(
            num_samples
        )

        vars_dir = tempdir / "vars_dir"
        write_vars(from_vcf, vars_dir)
        assert load_vars(
            vars_dir
        ).desired_num_vars_per_chunk == calc_num_vars_per_chunk(num_samples)
        assert (
            load_vars(vars_dir, desired_num_vars_per_chunk=2).desired_num_vars_per_chunk
            == 2
        )
