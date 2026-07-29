# Project Kaan Data Directory

This directory holds training audio organized by pest class. Follow these steps to prepare your dataset.

## Dataset Setup Instructions

### 1. Download IRRI Rice Acoustic Sensor Dataset

Download insect WAV files from the IRRI Rice Acoustic Sensor repository:

**https://github.com/cbalingbing/Rice-Acoustic-Sensor**

Clone or download the repository and locate the `Insect_wav_files` directory.

### 2. Download USDA Bug Bytes Dataset

Download stored-product insect acoustic recordings from USDA Ag Data Commons:

**https://agdatacommons.nal.usda.gov**

Search for **"Bug Bytes"** to find the acoustic library of stored-product insects.

### 3. Organize Rice Weevil Files

Place all `Sitophilus oryzae` WAV files in:

```
data/rice_weevil/
```

### 4. Organize Lesser Grain Borer Files

Place all `Rhyzopertha dominica` WAV files in:

```
data/lesser_grain_borer/
```

### 5. Organize Red Flour Beetle Files

Place all `Tribolium castaneum` WAV files in:

```
data/red_flour_beetle/
```

### 6. Add Clean (No Infestation) Audio

Record or source ambient grain storage sounds with no insect activity and place WAV files in:

```
data/clean/
```

**Tips for clean audio:**
- Record empty grain bins or hermetically sealed bags with no insects
- Capture ambient warehouse sounds without pest activity
- Use the same recording setup (phone against bag) for consistency

### 7. Train the Model

Once all directories contain WAV files, run:

```bash
python model/train.py
```

Then convert to TFLite for deployment:

```bash
python model/convert_tflite.py
```

## Expected Directory Structure

```
data/
  clean/                  *.wav: no insects
  rice_weevil/            *.wav: Sitophilus oryzae
  lesser_grain_borer/     *.wav: Rhyzopertha dominica
  red_flour_beetle/       *.wav: Tribolium castaneum
```

## Notes

- Supported format: WAV files at any sample rate (resampled to 16 kHz during training)
- Minimum recommended: 20+ files per class for meaningful training
- Audio is trimmed/padded to 10 seconds during preprocessing
- Augmentation (pitch shift, pink noise, time stretch) is applied to training data only
