from typing import Callable, Iterator, Protocol
import functools

from pynei.config import MAP_REDUCE_CHUNK_SIZE


class _ChunkProcessor:
    def __init__(self, map_functs):
        self.map_functs = map_functs

    def __call__(self, item):
        processed_item = item
        for one_funct in self.map_functs:
            processed_item = one_funct(processed_item)
        return processed_item


class Pipeline:
    def __init__(
        self,
        map_functs: list[Callable] | None = None,
        reduce_funct: Callable = None,
        reduce_initializer=None,
        after_reduce_funct: Callable = None,
    ):
        self.map_functs = map_functs
        self.reduce_funct = reduce_funct
        self.reduce_initializer = reduce_initializer
        self.after_reduce_funct = after_reduce_funct

    def append_map_funct(self, map_funct: Callable):
        self.map_functs.append(map_funct)

    def set_reduce_funct(self, reduce_funct: Callable, reduce_initializer=None):
        self.reduce_funct = reduce_funct
        self.reduce_initializer = reduce_initializer

    def _process_vars(
        self,
        variants,
        num_processes: int = 1,
        map_reduce_chunk_size=MAP_REDUCE_CHUNK_SIZE,
    ):
        process_chunk = _ChunkProcessor(self.map_functs)

        use_multiprocessing = num_processes > 1

        if use_multiprocessing:
            import threaded_map_reduce

            if self.reduce_funct:
                result = threaded_map_reduce.map_reduce(
                    map_fn=process_chunk,
                    reduce_fn=self.reduce_funct,
                    iterable=variants.iter_vars_chunks(),
                    num_computing_threads=num_processes,
                    chunk_size=map_reduce_chunk_size,
                )
            else:
                result = threaded_map_reduce.map(
                    map_fn=process_chunk,
                    items=variants.iter_vars_chunks(),
                    num_computing_threads=num_processes,
                    chunk_size=map_reduce_chunk_size,
                )
        else:
            processed_chunks = map(process_chunk, variants.iter_vars_chunks())
            result = processed_chunks
            if self.reduce_funct is not None:
                result = functools.reduce(
                    self.reduce_funct, processed_chunks, self.reduce_initializer
                )

        if self.after_reduce_funct is not None:
            result = self.after_reduce_funct(result)

        return result

    def map_chunks(self, variants, num_processes: int = 1) -> Iterator:
        if self.reduce_funct is not None or self.reduce_initializer is not None:
            raise ValueError(
                "For mapping reduce_funct and reduce_initializer must be None"
            )
        return self._process_vars(variants, num_processes)

    def map_and_reduce(self, variants, num_processes: int = 1):
        if self.reduce_funct is None:
            raise ValueError("For mapping and reducing reduce_funct must be set")
        return self._process_vars(variants, num_processes)


class ChunkCalc(Protocol):
    """One calculation done chunk by chunk, so that several can share a pass.

    calc_for_chunk gives the contribution of one chunk, in the same shape as
    the accumulated result, and reduce adds two of them. reduce has to be
    associative, because the threaded runs combine the contributions in an
    order of their own, and it is given None as the first accumulated value in
    the serial runs.
    """

    def calc_for_chunk(self, chunk, cache: dict): ...

    def reduce(self, accumulated, contribution): ...

    def finish(self, accumulated): ...


def run_chunk_calcs(
    variants, calcs: dict[str, ChunkCalc], num_processes: int = 1
) -> dict:
    """It runs several ChunkCalcs in one single pass over the variants.

    The cache given to calc_for_chunk lives for one chunk and it is shared by
    all the calcs done on that chunk, that is where they hand each other the
    intermediate results.
    """

    def calc_for_chunk(chunk):
        cache = {}
        return {name: calc.calc_for_chunk(chunk, cache) for name, calc in calcs.items()}

    def reduce(accumulated, contribution):
        if accumulated is None:
            return contribution
        return {
            name: calc.reduce(accumulated[name], contribution[name])
            for name, calc in calcs.items()
        }

    pipeline = Pipeline(map_functs=[calc_for_chunk], reduce_funct=reduce)
    accumulated = pipeline.map_and_reduce(variants, num_processes=num_processes)
    if accumulated is None:
        raise ValueError("There are no variants to calculate anything from")

    return {name: calc.finish(accumulated[name]) for name, calc in calcs.items()}
