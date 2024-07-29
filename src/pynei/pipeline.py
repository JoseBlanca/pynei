from typing import Callable, Iterator
import multiprocessing
import functools


class _ChunkProcessor:
    def __init__(self, map_functs):
        self.map_functs = map_functs

    def __call__(self, item):
        processed_item = item
        for one_funct in self.map_functs:
            processed_item = one_funct(processed_item)
        return processed_item


class Pipeline:
    def __init__(self):
        self.map_functs = []
        self.reduce_funct = None
        self.reduce_initializer = None
        self.num_processes = 1

    def append_map_funct(self, map_funct: Callable):
        self.map_functs.append(map_funct)

    def set_reduce_funct(self, reduce_funct: Callable, reduce_initializer=None):
        self.reduce_funct = reduce_funct
        self.reduce_initializer = reduce_initializer

    def _process_vars(self, vars, num_processes: int = 1):
        process_chunk = _ChunkProcessor(self.map_functs)

        use_multiprocessing = num_processes > 1

        if use_multiprocessing:
            pool = multiprocessing.Pool(3)
            map_ = pool.map
        else:
            map_ = map

        processed_chunks = map_(process_chunk, vars.iter_vars_chunks())

        if self.reduce_funct is not None:
            reduced_result = functools.reduce(
                self.reduce_funct, processed_chunks, self.reduce_initialializer
            )
            result = reduced_result
        else:
            result = processed_chunks

        if use_multiprocessing:
            pool.close()

        return result

    def map_chunks(self, vars, num_processes: int = 1) -> Iterator:
        if self.reduce_funct is not None or self.reduce_initializer is not None:
            raise ValueError(
                "For mapping reduce_funct and reduce_initializer must be None"
            )
        return self._process_vars(vars, num_processes)

    def map_and_reduce(self, vars, num_processes: int = 1):
        if self.reduce_funct is None:
            raise ValueError("For mapping and reducing reduce_funct must be set")
        return self._process_vars(vars, num_processes)
