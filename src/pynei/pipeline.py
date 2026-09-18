from typing import Callable, Iterator, Protocol
import functools
import queue
import threading

from pynei.config import MAP_REDUCE_CHUNK_SIZE, NUM_CHUNKS_READ_AHEAD


# put in the queue after the last chunk, so that the thread reading them can
# say that there are no more without closing anything
_NO_MORE_CHUNKS = object()


def _read_chunks_ahead(chunks, num_chunks_read_ahead=NUM_CHUNKS_READ_AHEAD):
    """It reads the chunks in a thread of its own, ahead of the work.

    Getting a chunk is mostly done outside of python, decompressing it in
    arrow or parsing the VCF, and those let go of the GIL, so the next chunk
    can be read while the one in hand is being worked on. It costs the memory
    of the chunks read ahead, one of them by default.

    One thread reads, so this hides the reading behind the work, it does not
    make the reading itself any faster. For that every thread would have to
    read its own chunk, which only a source that can seek to a chunk could do.
    """
    if not num_chunks_read_ahead:
        yield from chunks
        return

    buffer = queue.Queue(maxsize=num_chunks_read_ahead)

    def read_chunks():
        try:
            for chunk in chunks:
                buffer.put(chunk)
            buffer.put(_NO_MORE_CHUNKS)
        except queue.ShutDown:
            # whoever was asking for the chunks gave up before they were over
            pass
        except BaseException as error:
            # the error is raised again in the thread that asked for the chunks
            try:
                buffer.put(error)
            except queue.ShutDown:
                pass

    thread = threading.Thread(target=read_chunks, daemon=True)
    thread.start()
    try:
        while True:
            chunk = buffer.get()
            if chunk is _NO_MORE_CHUNKS:
                break
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk
    finally:
        # when the chunks are not asked for to the end, the thread reading
        # them would sit for ever on a queue that nobody empties
        buffer.shutdown(immediate=True)


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
        num_threads: int = 1,
        map_reduce_chunk_size=MAP_REDUCE_CHUNK_SIZE,
        num_chunks_read_ahead=NUM_CHUNKS_READ_AHEAD,
    ):
        process_chunk = _ChunkProcessor(self.map_functs)

        chunks = _read_chunks_ahead(variants.iter_vars_chunks(), num_chunks_read_ahead)

        use_threads = num_threads > 1

        if use_threads:
            import threaded_map_reduce

            if self.reduce_funct:
                result = threaded_map_reduce.map_reduce(
                    map_fn=process_chunk,
                    reduce_fn=self.reduce_funct,
                    iterable=chunks,
                    num_computing_threads=num_threads,
                    chunk_size=map_reduce_chunk_size,
                )
            else:
                result = threaded_map_reduce.map(
                    map_fn=process_chunk,
                    items=chunks,
                    num_computing_threads=num_threads,
                    chunk_size=map_reduce_chunk_size,
                )
        else:
            processed_chunks = map(process_chunk, chunks)
            result = processed_chunks
            if self.reduce_funct is not None:
                result = functools.reduce(
                    self.reduce_funct, processed_chunks, self.reduce_initializer
                )

        if self.after_reduce_funct is not None:
            result = self.after_reduce_funct(result)

        return result

    def map_chunks(self, variants, num_threads: int = 1) -> Iterator:
        if self.reduce_funct is not None or self.reduce_initializer is not None:
            raise ValueError(
                "For mapping reduce_funct and reduce_initializer must be None"
            )
        # threaded_map_reduce.map keeps the order of the items, so the chunks
        # come out in the order of the variants however many threads run
        return self._process_vars(variants, num_threads)

    def map_and_reduce(self, variants, num_threads: int = 1):
        if self.reduce_funct is None:
            raise ValueError("For mapping and reducing reduce_funct must be set")
        return self._process_vars(variants, num_threads)


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
    variants, calcs: dict[str, ChunkCalc], num_threads: int = 1
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
    accumulated = pipeline.map_and_reduce(variants, num_threads=num_threads)
    if accumulated is None:
        raise ValueError("There are no variants to calculate anything from")

    return {name: calc.finish(accumulated[name]) for name, calc in calcs.items()}
