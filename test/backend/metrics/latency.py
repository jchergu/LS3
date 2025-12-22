import time

def measure_latency(fn, *args, runs=10, **kwargs):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": sum(times) / len(times),
        "p95_ms": sorted(times)[int(0.95 * len(times))],
    }
