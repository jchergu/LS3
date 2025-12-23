# Comparisons for vector index
This folder contains code to compare different implementations of the vector index:
- NumPy (brute-force)
- FAISS

You can add/edit queries in the `comparison/queries.txt` file.

## How to run comparison

```bash
python -m comparison.run_benchmark
```

Results are stored in `results.csv` file and plots in `plots/` folder.

## Implementations compared

As we will see, the FAISS implementation is significantly faster than the NumPy one.
### Latency boxplot
[![](plots/latency_boxplot.png)](plots/latency_boxplot.png)

### Latency comparison
[![](plots/latency_comparison.png)](plots/latency_comparison.png)

          numpy_ms    faiss_ms
count     4.000000    4.000000
mean   2476.312415  220.433960
std     182.932186   37.074608
min    2248.863640  169.068600
25%    2394.425020  205.367430
50%    2483.702230  232.316020
75%    2565.589625  247.382550
max    2688.981560  248.035200