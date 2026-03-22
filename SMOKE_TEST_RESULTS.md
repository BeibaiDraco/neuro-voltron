# Smoke Test Results

These results were obtained locally on CPU with small synthetic datasets
(24 trials, 3 regions, 1 training epoch per variant).

They are intended only as a sanity check that the package runs end-to-end,
not as scientific benchmarks.

| Variant | Scenario | Best val loss | Effectome cosine |
|---|---|---:|---:|
| A0 | additive | 0.172446 | 0.683019 |
| A1 | additive | 0.172580 | 0.683064 |
| A2 | additive | 0.205390 | 0.692726 |
| B1 | additive | 0.311285 | 0.575812 |
| B2 | additive | 0.414934 | 0.741970 |
| B3 | additive | 0.371544 | 0.575744 |
| C1 | modulatory | 0.201901 | 0.767526 |
| C2 | modulatory | 0.207694 | 0.686793 |

Each smoke run verified:

- synthetic data generation,
- dataset loading,
- baseline handling,
- model construction,
- forward pass,
- one-epoch optimization,
- artifact saving,
- effectome computation.
