import threading
import time

import numpy
import pytest

from pynei.pipeline import Pipeline, _read_chunks_ahead
from pynei.variants import Variants, Genotypes, VariantsChunk
from .var_generators import create_sample_names


def _add(accumulated, contribution):
    # the serial runs give None as the first accumulated value
    if accumulated is None:
        return contribution
    return accumulated + contribution


def _create_chunk(num_vars=2, num_samples=4, ploidy=2):
    gt_array = numpy.ma.array(
        numpy.random.randint(0, 2, size=(num_vars, num_samples, ploidy))
    )
    return VariantsChunk(gts=Genotypes(gt_array, samples=create_sample_names(gt_array)))


class _RecordingFactory:
    """It says in which thread every chunk was read."""

    def __init__(self, num_chunks=4, delay=0.0):
        self.num_chunks = num_chunks
        self.delay = delay
        self.reading_threads = []

    def _get_metadata(self):
        chunk = _create_chunk()
        return {
            "samples": chunk.gts.samples,
            "num_samples": chunk.num_samples,
            "ploidy": chunk.ploidy,
        }

    def iter_vars_chunks(self):
        for _ in range(self.num_chunks):
            if self.delay:
                time.sleep(self.delay)
            self.reading_threads.append(threading.current_thread())
            yield _create_chunk()


def test_the_chunks_come_out_whole_and_in_order():
    chunks = [_create_chunk(num_vars=idx + 1) for idx in range(5)]
    read = list(_read_chunks_ahead(iter(chunks), 1))
    assert [chunk.num_vars for chunk in read] == [1, 2, 3, 4, 5]
    assert read == chunks


def test_reading_none_ahead_gives_the_same_chunks():
    chunks = [_create_chunk(num_vars=idx + 1) for idx in range(5)]
    assert list(_read_chunks_ahead(iter(chunks), 0)) == chunks


def test_the_chunks_are_read_in_another_thread():
    factory = _RecordingFactory(num_chunks=4)
    # the size asked for is the one the source gives, so that the chunks go
    # through one by one instead of being buffered and joined by _resize_chunks
    variants = Variants(factory, desired_num_vars_per_chunk=2)
    working_threads = []

    def work(chunk):
        working_threads.append(threading.current_thread())
        return chunk.num_vars

    pipeline = Pipeline(map_functs=[work], reduce_funct=_add)
    total = pipeline.map_and_reduce(variants, num_threads=1)

    assert total == 8
    assert len(factory.reading_threads) == 4
    # the reading is done by one thread of its own, and the work by another
    reading = set(factory.reading_threads)
    assert len(reading) == 1
    assert reading.isdisjoint(set(working_threads))
    assert threading.current_thread() in set(working_threads)


def test_the_reading_runs_ahead_of_the_work():
    """The point of it: while a chunk is worked on the next one is read."""
    factory = _RecordingFactory(num_chunks=4)
    variants = Variants(factory, desired_num_vars_per_chunk=2)
    chunks_read_when_work_started = []

    def work(chunk):
        chunks_read_when_work_started.append(len(factory.reading_threads))
        time.sleep(0.02)
        return 1

    pipeline = Pipeline(map_functs=[work], reduce_funct=_add)
    pipeline.map_and_reduce(variants, num_threads=1)

    # by the time the first chunk is worked on the second one has been read
    assert chunks_read_when_work_started[0] >= 2


def test_an_error_while_reading_is_raised_to_whoever_asked():
    def chunks():
        yield _create_chunk()
        raise ValueError("the source is broken")

    with pytest.raises(ValueError, match="the source is broken"):
        list(_read_chunks_ahead(chunks(), 1))


def test_giving_up_early_does_not_leave_the_reading_thread_stuck():
    threads_before = threading.active_count()

    def many_chunks():
        while True:
            yield _create_chunk()

    reader = _read_chunks_ahead(many_chunks(), 1)
    next(reader)
    next(reader)
    reader.close()

    # the thread reading them is let go, it does not sit on a queue for ever
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and threading.active_count() > threads_before:
        time.sleep(0.01)
    assert threading.active_count() == threads_before


def test_the_filtering_stats_are_complete_although_the_reading_runs_ahead():
    from pynei import filter_by_missing_data, gather_filtering_stats
    from pynei.per_var_stats import calc_per_var_distribs

    gts = numpy.array(
        [
            [[0, 0], [0, 1], [0, 0], [0, 0], [1, 1]],
            [[-1, -1], [-1, -1], [0, 0], [0, 1], [1, 1]],
            [[0, 0], [0, 1], [1, 1], [0, 0], [0, 1]],
        ]
    )
    variants = Variants.from_gt_array(gts, samples=[f"s{idx}" for idx in range(5)])
    variants = filter_by_missing_data(variants, max_allowed_missing_rate=0.1)
    calc_per_var_distribs(variants)
    stats = gather_filtering_stats(variants)
    assert stats["missing_data"].vars_processed == 3
    assert stats["missing_data"].vars_kept == 2


def test_the_queue_is_bounded_so_the_reading_does_not_run_away():
    """It reads ahead, it does not read everything into memory."""
    reading = []

    def chunks():
        for idx in range(20):
            reading.append(idx)
            yield _create_chunk()

    reader = _read_chunks_ahead(chunks(), 1)
    next(reader)
    time.sleep(0.05)
    # one chunk in hand, one in the queue and one waiting to be put in it
    assert len(reading) <= 3
    reader.close()
