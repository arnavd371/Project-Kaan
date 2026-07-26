# Dataset Download Instructions

The WAV training files are not included due to size.

## IRRI Rice Acoustic Sensor Dataset
git clone https://github.com/cbalingbing/Rice-Acoustic-Sensor
Unzip Insect_WAVs.zip and copy:
- S_oryzae/*.wav → data/rice_weevil/
- R_dominica/*.wav → data/lesser_grain_borer/
- T_castaneum/*.wav → data/red_flour_beetle/

## Clean Class
Run: python model/generate_clean_audio.py
This generates 500 pink noise files for the clean class.

## Then train
python model/train.py
python model/convert_tflite.py

Note: project-kaan.tflite is included so the app works
immediately without retraining.
