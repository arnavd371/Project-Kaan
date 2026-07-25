# Kaan | कान
## AI-Powered Acoustic Grain Pest Detector for Indian Farmers

### The Problem
India loses approximately Rs 1,300 crore annually to insect
infestations in stored grain (IGMRI, Ministry of Food and
Public Distribution, 2015). Rice weevil, lesser grain borer,
and red flour beetle are the three primary culprits. By the
time infestation is visible, 10-20% of stored grain is already
damaged. No affordable early warning tool exists for the 100
million small and marginal farmers who store grain at home.

### The Solution
Kaan (Hindi: कान, meaning "ear") is a free, offline smartphone
app. A farmer places their phone against a grain storage bag
and records 30 seconds of audio. An on-device CNN classifies
the recording and delivers an advisory in the farmer's language
within seconds — no internet required.

### Model Performance
- Training data: IRRI Rice Acoustic Sensor Dataset
  (Balingbing et al., Computers & Electronics in Agriculture, 2024)
- Best validation accuracy: 97.76% (macro F1 0.98; leakage-fixed + SpecAugment/deeper CNN, audited)
- Exceeds reference paper accuracy of 84.51%
- Model size: 333 KB (INT8 quantized TFLite)
- Classes: clean, rice weevil, lesser grain borer, red flour beetle

### Novel Contribution
First phone-microphone, offline, Indian-language acoustic
grain pest detection tool. All existing systems require
dedicated hardware (Raspberry Pi + MEMS microphone).
Kaan works on any basic Android or iOS phone.

### Features
- 5 Indian languages: English, Hindi, Marathi, Punjabi, Telugu
- Fully offline — no internet required
- Severity estimation from acoustic signal density
- Confidence threshold — says uncertain rather than wrong
- Multilingual farmer advisories with specific actions
- Free, no registration, no hardware purchase

### Tech Stack
Python, TensorFlow Lite, librosa, mel spectrogram CNN,
Streamlit, INT8 quantization

### SDGs
SDG 2 (Zero Hunger), SDG 12 (Responsible Consumption)

### Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Dataset
See [data/HOW_TO_GET_DATA.md](data/HOW_TO_GET_DATA.md)

### Citation
Balingbing C et al. (2024). Application of a multi-layer CNN
to classify major insect pests in stored rice detected by an
acoustic device. Computers and Electronics in Agriculture,
225, 109297.

### License
MIT License. Copyright (c) 2026 Arnav Dhiman. See [LICENSE](LICENSE).

Pest training audio is from the open IRRI Rice Acoustic Sensor dataset
(Balingbing et al.): https://github.com/cbalingbing/Rice-Acoustic-Sensor
