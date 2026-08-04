# Findings

## Headlines

- Best same-split accuracy: `gbdt` at 0.9589 (macro-F1 0.9671); beats ref=True.
- Best classical `gbdt` (0.9589) vs best CNN `cnn_deep` (0.9557); delta=+0.0032.
- McNemar classical vs CNN: discordant=15, p=1.

## Efficiency

| Approach | Acc | Macro F1 | Params | Train s |
|---|---:|---:|---:|---:|
| `gbdt` | 0.9589 | 0.9671 | - | 2.2 |
| `svm_rbf` | 0.9557 | 0.9648 | - | 0.1 |
| `cnn_deep` | 0.9557 | 0.9647 | 313764 | 147.2 |
| `extratrees` | 0.9557 | 0.9641 | - | 1.0 |
| `cnn_shallow` | 0.9430 | 0.9545 | 110596 | 69.1 |
| `logreg` | 0.9430 | 0.9544 | 300 | 0.1 |
| `rf` | 0.9367 | 0.9488 | - | 2.4 |
| `mlp` | 0.9177 | 0.9338 | 18116 | 0.3 |
| `knn` | 0.9051 | 0.9232 | - | 0.0 |
| `yamnet_probe` | 0.8671 | 0.8937 | 4100 | 20.3 |
| `cnn1d` | 0.5823 | 0.6470 | 149188 | 19.8 |

## McNemar (lowest p)

- `cnn1d` vs `svm_rbf`: p=1.821e-34, discordant=120 (b=1, c=119)
- `cnn1d` vs `extratrees`: p=2.823e-33, discordant=122 (b=2, c=120)
- `cnn1d` vs `gbdt`: p=8.422e-31, discordant=129 (b=5, c=124)
- `cnn_deep` vs `cnn1d`: p=1.62e-30, discordant=128 (b=123, c=5)
- `cnn1d` vs `rf`: p=1.649e-30, discordant=118 (b=3, c=115)
- `cnn_shallow` vs `cnn1d`: p=3.418e-30, discordant=122 (b=118, c=4)
- `cnn1d` vs `logreg`: p=3.418e-30, discordant=122 (b=4, c=118)
- `cnn1d` vs `mlp`: p=6.664e-28, discordant=114 (b=4, c=110)
- `cnn1d` vs `knn`: p=1.198e-24, discordant=116 (b=7, c=109)
- `cnn1d` vs `yamnet_probe`: p=1.817e-17, discordant=122 (b=16, c=106)
- `yamnet_probe` vs `extratrees`: p=7.66e-07, discordant=34 (b=3, c=31)
- `yamnet_probe` vs `svm_rbf`: p=4.256e-06, discordant=38 (b=5, c=33)

## Top confusions

- `cnn_shallow`: 18/316 errors; lesser_grain_borer→rice_weevil (11), rice_weevil→lesser_grain_borer (3), rice_weevil→red_flour_beetle (3), red_flour_beetle→rice_weevil (1)
- `cnn_deep`: 14/316 errors; rice_weevil→lesser_grain_borer (6), lesser_grain_borer→rice_weevil (5), lesser_grain_borer→red_flour_beetle (1), red_flour_beetle→rice_weevil (1)
- `cnn1d`: 132/316 errors; lesser_grain_borer→red_flour_beetle (45), lesser_grain_borer→rice_weevil (28), rice_weevil→red_flour_beetle (17), red_flour_beetle→lesser_grain_borer (16)
- `yamnet_probe`: 42/316 errors; lesser_grain_borer→rice_weevil (19), rice_weevil→lesser_grain_borer (14), rice_weevil→red_flour_beetle (5), red_flour_beetle→rice_weevil (3)
- `svm_rbf`: 14/316 errors; lesser_grain_borer→rice_weevil (9), rice_weevil→lesser_grain_borer (3), rice_weevil→red_flour_beetle (1), red_flour_beetle→rice_weevil (1)
- `mlp`: 26/316 errors; lesser_grain_borer→rice_weevil (11), rice_weevil→lesser_grain_borer (6), rice_weevil→red_flour_beetle (6), red_flour_beetle→rice_weevil (2)
- `gbdt`: 13/316 errors; rice_weevil→lesser_grain_borer (5), lesser_grain_borer→rice_weevil (4), red_flour_beetle→lesser_grain_borer (2), rice_weevil→red_flour_beetle (1)
- `rf`: 20/316 errors; lesser_grain_borer→rice_weevil (6), rice_weevil→red_flour_beetle (5), rice_weevil→lesser_grain_borer (4), lesser_grain_borer→red_flour_beetle (3)
- `extratrees`: 14/316 errors; rice_weevil→red_flour_beetle (6), lesser_grain_borer→rice_weevil (4), rice_weevil→lesser_grain_borer (2), lesser_grain_borer→red_flour_beetle (2)
- `knn`: 30/316 errors; lesser_grain_borer→rice_weevil (10), rice_weevil→red_flour_beetle (8), rice_weevil→lesser_grain_borer (6), lesser_grain_borer→red_flour_beetle (3)
- `logreg`: 18/316 errors; rice_weevil→lesser_grain_borer (7), lesser_grain_borer→rice_weevil (7), rice_weevil→red_flour_beetle (2), lesser_grain_borer→red_flour_beetle (1)

## SNR proxy

- Clean probe acc: 0.8211
- SNR 20.0 dB: probe_acc=0.8421 (drop=-0.0211)
- SNR 10.0 dB: probe_acc=0.7474 (drop=+0.0737)
- SNR 5.0 dB: probe_acc=0.6947 (drop=+0.1263)
- SNR 0.0 dB: probe_acc=0.6105 (drop=+0.2105)
- Proxy only; not a phone-mic field test.

## INT8 vs float

- Skipped: production H5/TFLite missing in this environment

## Notes

- McNemar is single-fold; report multi-seed CIs with it.
- SNR curve is a proxy, not a phone-mic field study.
- Reference 84.51% is the cited Balingbing number on this protocol, not a locked reimplementation.

