# Findings

## Headlines

- Best same-split accuracy: `cnn_deep` at 0.9462 (macro-F1 0.9530); beats ref=True.
- Best classical `gbdt` (0.9367) vs best CNN `cnn_deep` (0.9462); delta=-0.0095.
- McNemar classical vs CNN: discordant=17, p=0.6291.

## Efficiency

| Approach | Acc | Macro F1 | Params | Train s |
|---|---:|---:|---:|---:|
| `cnn_deep` | 0.9462 | 0.9530 | 313764 | 141.2 |
| `cnn_shallow` | 0.9430 | 0.9504 | 110596 | 65.9 |
| `gbdt` | 0.9367 | 0.9452 | - | 2.2 |
| `extratrees` | 0.9367 | 0.9452 | - | 1.0 |
| `svm_rbf` | 0.9335 | 0.9391 | - | 0.1 |
| `logreg` | 0.9241 | 0.9387 | 300 | 0.1 |
| `rf` | 0.9272 | 0.9374 | - | 2.4 |
| `mlp` | 0.9209 | 0.9363 | 18116 | 0.3 |
| `knn` | 0.9051 | 0.9196 | - | 0.0 |
| `yamnet_probe` | 0.8671 | 0.8891 | 4100 | 20.6 |
| `cnn1d` | 0.4810 | 0.5325 | 149188 | 16.5 |

## McNemar (lowest p)

- `cnn1d` vs `extratrees`: p=8.968e-44, discordant=144 (b=0, c=144)
- `cnn1d` vs `logreg`: p=9.364e-40, discordant=144 (b=2, c=142)
- `cnn_deep` vs `cnn1d`: p=1.04e-39, discordant=155 (b=151, c=4)
- `cnn1d` vs `gbdt`: p=7.693e-39, discordant=152 (b=4, c=148)
- `cnn1d` vs `mlp`: p=2.279e-38, discordant=145 (b=3, c=142)
- `cnn1d` vs `rf`: p=5.681e-38, discordant=149 (b=4, c=145)
- `cnn_shallow` vs `cnn1d`: p=1.118e-37, discordant=158 (b=152, c=6)
- `cnn1d` vs `svm_rbf`: p=1.185e-37, discordant=153 (b=5, c=148)
- `cnn1d` vs `knn`: p=5.995e-36, discordant=142 (b=4, c=138)
- `cnn1d` vs `yamnet_probe`: p=9.448e-27, discordant=148 (b=13, c=135)
- `cnn_deep` vs `yamnet_probe`: p=1.093e-05, discordant=33 (b=29, c=4)
- `cnn_shallow` vs `yamnet_probe`: p=3.856e-05, discordant=34 (b=29, c=5)

## Top confusions

- `cnn_shallow`: 18/316 errors; lesser_grain_borer→rice_weevil (8), rice_weevil→lesser_grain_borer (3), rice_weevil→red_flour_beetle (3), red_flour_beetle→lesser_grain_borer (2)
- `cnn_deep`: 17/316 errors; lesser_grain_borer→rice_weevil (8), rice_weevil→lesser_grain_borer (4), rice_weevil→red_flour_beetle (3), red_flour_beetle→clean (1)
- `cnn1d`: 164/316 errors; lesser_grain_borer→red_flour_beetle (71), rice_weevil→red_flour_beetle (51), lesser_grain_borer→rice_weevil (24), red_flour_beetle→rice_weevil (11)
- `yamnet_probe`: 42/316 errors; rice_weevil→lesser_grain_borer (18), lesser_grain_borer→rice_weevil (14), rice_weevil→red_flour_beetle (6), red_flour_beetle→lesser_grain_borer (2)
- `svm_rbf`: 21/316 errors; rice_weevil→lesser_grain_borer (7), lesser_grain_borer→rice_weevil (7), rice_weevil→red_flour_beetle (3), red_flour_beetle→rice_weevil (2)
- `mlp`: 25/316 errors; lesser_grain_borer→rice_weevil (10), rice_weevil→lesser_grain_borer (7), rice_weevil→red_flour_beetle (6), red_flour_beetle→lesser_grain_borer (2)
- `gbdt`: 20/316 errors; lesser_grain_borer→rice_weevil (7), rice_weevil→lesser_grain_borer (6), rice_weevil→red_flour_beetle (4), red_flour_beetle→clean (1)
- `rf`: 23/316 errors; rice_weevil→lesser_grain_borer (9), lesser_grain_borer→rice_weevil (7), rice_weevil→red_flour_beetle (5), red_flour_beetle→clean (1)
- `extratrees`: 20/316 errors; lesser_grain_borer→rice_weevil (9), rice_weevil→lesser_grain_borer (5), rice_weevil→red_flour_beetle (5), red_flour_beetle→clean (1)
- `knn`: 30/316 errors; lesser_grain_borer→rice_weevil (12), rice_weevil→lesser_grain_borer (8), rice_weevil→red_flour_beetle (6), lesser_grain_borer→red_flour_beetle (1)
- `logreg`: 24/316 errors; lesser_grain_borer→rice_weevil (11), rice_weevil→red_flour_beetle (7), rice_weevil→lesser_grain_borer (6)

## SNR proxy

- Clean probe acc: 0.8842
- SNR 20.0 dB: probe_acc=0.9158 (drop=-0.0316)
- SNR 10.0 dB: probe_acc=0.7263 (drop=+0.1579)
- SNR 5.0 dB: probe_acc=0.7474 (drop=+0.1368)
- SNR 0.0 dB: probe_acc=0.6000 (drop=+0.2842)
- Note: Proxy robustness only. Phone-mic external set remains the gold domain-shift test.

## INT8 vs float

- Skipped: production H5/TFLite missing in this environment

## Notes

- McNemar is paired on the same file-level val fold; multi-seed CIs still matter more than single-seed p.
- SNR curve is a proxy, not a phone-mic field study.
- Soft claim vs Balingbing 84.51%: cited number on own protocol, not locked reimplementation.

