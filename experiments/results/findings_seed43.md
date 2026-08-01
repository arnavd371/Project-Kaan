# Findings

## Headlines

- Best same-split accuracy: `gbdt` at 0.9652 (macro-F1 0.9682); beats ref=True.
- Best classical `gbdt` (0.9652) vs best CNN `cnn_deep` (0.9525); delta=+0.0127.
- McNemar classical vs CNN: discordant=18, p=0.4807.

## Efficiency

| Approach | Acc | Macro F1 | Params | Train s |
|---|---:|---:|---:|---:|
| `gbdt` | 0.9652 | 0.9682 | - | 2.1 |
| `logreg` | 0.9557 | 0.9648 | 300 | 0.1 |
| `extratrees` | 0.9557 | 0.9609 | - | 1.0 |
| `svm_rbf` | 0.9525 | 0.9583 | - | 0.1 |
| `cnn_deep` | 0.9525 | 0.9581 | 313764 | 139.1 |
| `rf` | 0.9462 | 0.9530 | - | 2.3 |
| `mlp` | 0.9241 | 0.9393 | 18116 | 0.2 |
| `cnn_shallow` | 0.9209 | 0.9327 | 110596 | 63.0 |
| `knn` | 0.8987 | 0.9191 | - | 0.0 |
| `cnn1d` | 0.8797 | 0.9038 | 149188 | 27.1 |
| `yamnet_probe` | 0.8354 | 0.8657 | 4100 | 18.8 |

## McNemar (lowest p)

- `yamnet_probe` vs `gbdt`: p=2.328e-09, discordant=51 (b=5, c=46)
- `yamnet_probe` vs `extratrees`: p=3.244e-08, discordant=50 (b=6, c=44)
- `yamnet_probe` vs `logreg`: p=6.974e-08, discordant=52 (b=7, c=45)
- `cnn_deep` vs `yamnet_probe`: p=1.212e-07, discordant=51 (b=44, c=7)
- `yamnet_probe` vs `svm_rbf`: p=1.212e-07, discordant=51 (b=7, c=44)
- `yamnet_probe` vs `rf`: p=1.221e-06, discordant=53 (b=9, c=44)
- `cnn1d` vs `gbdt`: p=3.465e-06, discordant=35 (b=4, c=31)
- `gbdt` vs `knn`: p=1.943e-05, discordant=25 (b=23, c=2)
- `cnn1d` vs `extratrees`: p=6.96e-05, discordant=36 (b=6, c=30)
- `cnn1d` vs `logreg`: p=6.96e-05, discordant=36 (b=6, c=30)
- `extratrees` vs `knn`: p=0.0001211, discordant=22 (b=20, c=2)
- `cnn_deep` vs `cnn1d`: p=0.0001911, discordant=37 (b=30, c=7)

## Top confusions

- `cnn_shallow`: 25/316 errors; lesser_grain_borer→rice_weevil (15), red_flour_beetle→lesser_grain_borer (4), red_flour_beetle→rice_weevil (3), rice_weevil→lesser_grain_borer (1)
- `cnn_deep`: 15/316 errors; rice_weevil→lesser_grain_borer (6), lesser_grain_borer→rice_weevil (5), rice_weevil→red_flour_beetle (2), red_flour_beetle→clean (1)
- `cnn1d`: 38/316 errors; rice_weevil→lesser_grain_borer (17), red_flour_beetle→rice_weevil (9), lesser_grain_borer→rice_weevil (6), red_flour_beetle→lesser_grain_borer (5)
- `yamnet_probe`: 52/316 errors; lesser_grain_borer→rice_weevil (22), rice_weevil→lesser_grain_borer (20), red_flour_beetle→rice_weevil (5), rice_weevil→red_flour_beetle (2)
- `svm_rbf`: 15/316 errors; lesser_grain_borer→rice_weevil (9), rice_weevil→lesser_grain_borer (3), rice_weevil→red_flour_beetle (1), red_flour_beetle→clean (1)
- `mlp`: 24/316 errors; lesser_grain_borer→rice_weevil (12), rice_weevil→lesser_grain_borer (3), rice_weevil→red_flour_beetle (3), red_flour_beetle→rice_weevil (3)
- `gbdt`: 11/316 errors; lesser_grain_borer→rice_weevil (5), rice_weevil→red_flour_beetle (2), red_flour_beetle→rice_weevil (2), rice_weevil→lesser_grain_borer (1)
- `rf`: 17/316 errors; lesser_grain_borer→rice_weevil (9), rice_weevil→red_flour_beetle (3), red_flour_beetle→rice_weevil (2), rice_weevil→lesser_grain_borer (1)
- `extratrees`: 14/316 errors; lesser_grain_borer→rice_weevil (9), rice_weevil→lesser_grain_borer (2), rice_weevil→red_flour_beetle (1), red_flour_beetle→clean (1)
- `knn`: 32/316 errors; lesser_grain_borer→rice_weevil (16), rice_weevil→lesser_grain_borer (5), red_flour_beetle→rice_weevil (4), rice_weevil→red_flour_beetle (3)
- `logreg`: 14/316 errors; lesser_grain_borer→rice_weevil (7), rice_weevil→lesser_grain_borer (4), red_flour_beetle→lesser_grain_borer (2), red_flour_beetle→rice_weevil (1)

## SNR proxy

- Clean probe acc: 0.8632
- SNR 20.0 dB: probe_acc=0.8316 (drop=+0.0316)
- SNR 10.0 dB: probe_acc=0.7263 (drop=+0.1368)
- SNR 5.0 dB: probe_acc=0.6842 (drop=+0.1789)
- SNR 0.0 dB: probe_acc=0.6105 (drop=+0.2526)
- Note: Proxy robustness only. Phone-mic external set remains the gold domain-shift test.

## INT8 vs float

- Skipped: production H5/TFLite missing in this environment

## Notes

- McNemar is paired on the same file-level val fold; multi-seed CIs still matter more than single-seed p.
- SNR curve is a proxy, not a phone-mic field study.
- Soft claim vs Balingbing 84.51%: cited number on own protocol, not locked reimplementation.

