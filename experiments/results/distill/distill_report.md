# Distillation report

- Seed: `42`
- Teachers: `gbdt, extratrees, cnn_deep`
- T=2.0, alpha(hard)=0.5
- Ensemble val acc: **95.8861%** (macro F1 0.9670)
- Hard-only deep val acc: **95.5696%** (macro F1 0.9649)
- Distilled student val acc: **97.1519%** (macro F1 0.9771)
- Shipped: **distilled** (val acc 97.1519%)
- Params: 313764
- Elapsed: 497.8s

Next: `python -m model.export_deploy`
