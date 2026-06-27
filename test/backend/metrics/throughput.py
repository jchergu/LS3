import time

def measure_throughput(fn, queries):
    start = time.perf_counter()
    for q in queries:
        fn(q)
    elapsed = time.perf_counter() - start
    return len(queries) / elapsed
