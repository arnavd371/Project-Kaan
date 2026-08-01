# Statistical summary

Reference accuracy (Balingbing et al. 2024): **84.51%**

Tests (per approach, across seeds):
- Mean ± std accuracy / macro-F1
- Bootstrap 95% CI of the mean (10k resamples)
- One-sided one-sample t-test: H1 mean accuracy > reference
- One-sided Wilcoxon signed-rank on (acc − reference)
- Fraction of seeds beating reference (Wilson 95% CI)

| Approach | Acc mean±std | Acc 95% CI | Macro-F1 mean±std | Seeds > ref | t p (>) | Wilcoxon p (>) | CI > ref |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `gbdt` | 0.9536±0.0150 | [0.9367, 0.9652] | 0.9602±0.0130 | 3/3 | 0.003137 | 0.125 | yes |
| `cnn_deep` | 0.9515±0.0048 | [0.9462, 0.9557] | 0.9586±0.0059 | 3/3 | 0.0003438 | 0.125 | yes |
| `extratrees` | 0.9494±0.0110 | [0.9367, 0.9557] | 0.9567±0.0101 | 3/3 | 0.001832 | 0.125 | yes |
| `svm_rbf` | 0.9473±0.0120 | [0.9335, 0.9557] | 0.9541±0.0134 | 3/3 | 0.002277 | 0.125 | yes |
| `logreg` | 0.9409±0.0159 | [0.9241, 0.9557] | 0.9527±0.0131 | 3/3 | 0.004542 | 0.125 | yes |
| `rf` | 0.9367±0.0095 | [0.9272, 0.9462] | 0.9464±0.0081 | 3/3 | 0.00178 | 0.125 | yes |
| `cnn_shallow` | 0.9357±0.0128 | [0.9209, 0.9430] | 0.9459±0.0116 | 3/3 | 0.003292 | 0.125 | yes |
| `mlp` | 0.9209±0.0032 | [0.9177, 0.9241] | 0.9365±0.0027 | 3/3 | 0.0002903 | 0.125 | yes |
| `knn` | 0.9030±0.0037 | [0.8987, 0.9051] | 0.9206±0.0023 | 3/3 | 0.0006636 | 0.125 | yes |
| `yamnet_probe` | 0.8565±0.0183 | [0.8354, 0.8671] | 0.8828±0.0150 | 2/3 | 0.1957 | 0.25 | no |
| `cnn1d` | 0.6477±0.2073 | [0.4810, 0.8797] | 0.6944±0.1901 | 1/3 | 0.8796 | 0.875 | no |

Notes:
- With few seeds, p-values are underpowered; report CIs alongside tests.
- `CI > ref` means the bootstrap 95% CI lower bound exceeds the reference.
- Comparisons to Balingbing et al. use the cited accuracy number, not a shared reimplementation.

