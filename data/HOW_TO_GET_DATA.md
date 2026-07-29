# Dataset Download Instructions

The WAV training files are not included due to size.

## Kaggle (recommended for full IRRI + GPU benchmark)

See `experiments/KAGGLE.md`. The kernel
`arnavd371/kaan-multi-approach-benchmark` prepares data the same way as
`model/train_kaggle.py` (IRRI clone + Speech Commands clean windows) and runs
the multi-approach bake-off.

```bash
bash experiments/kaggle/push_and_run.sh
kaggle kernels status arnavd371/kaan-multi-approach-benchmark
```

## IRRI Rice Acoustic Sensor Dataset (local)
git clone https://github.com/cbalingbing/Rice-Acoustic-Sensor
Unzip Insect_WAVs.zip and copy:
- S_oryzae/*.wav → data/rice_weevil/
- R_dominica/*.wav → data/lesser_grain_borer/
- T_castaneum/*.wav → data/red_flour_beetle/

Or one-shot (needs network):

```bash
python -m experiments.prepare_kaggle_data --out .
```

## Clean Class
Run: python model/generate_clean_audio.py
This generates 500 pink noise files for the clean class.

(On Kaggle / `prepare_kaggle_data`, clean comes from Speech Commands
`_background_noise_` 10 s windows: same source as `model/train_kaggle.py`.)

## Then train
python model/train.py
python model/convert_tflite.py

Note: project-kaan.tflite is included so the app works
immediately without retraining.
