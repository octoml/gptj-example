import dataclasses
import typing
import time


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    latencies_ms: typing.Tuple[float, ...]
    start_ns: int
    end_ns: int

    @property
    def total_duration_ms(self) -> float:
        return float((self.end_ns - self.start_ns) / 1e6)

    @property
    def qps(self) -> float:
        duration_secs = self.total_duration_ms / 1e3
        return len(self.latencies_ms) / duration_secs


def benchmark_fn(
    fn: typing.Callable,
    num_warmups: int,
    num_runs: int,
) -> BenchmarkResult:
    for _ in range(num_warmups):
        fn()
    latencies_ms = []
    benchmark_start_ns = start_ns = time.perf_counter_ns()
    for i in range(num_runs):
        fn()
        end_ns = time.perf_counter_ns()
        elapsed_ms = float(end_ns - start_ns) / 1e6
        latencies_ms.append(elapsed_ms)
    benchmark_end_ns = end_ns
    return BenchmarkResult(
        latencies_ms=latencies_ms, start_ns=benchmark_start_ns, end_ns=benchmark_end_ns
    )
