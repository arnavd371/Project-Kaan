# Kaan | कान
## AI-Powered Acoustic Grain Pest Detector for Indian Farmers

### One-Line Summary
Kaan listens to grain so farmers do not have to wait until they can see the damage.

### The Problem (Metric 1: Impact Evidence)
India stores over 80 million tonnes of food grain. Insects cause approximately 1,300 crore rupees in annual storage losses (IGMRI, Ministry of Food and Public Distribution, 2015). The three primary culprits are rice weevil (Sitophilus oryzae), lesser grain borer (Rhyzopertha dominica), and red flour beetle (Tribolium castaneum). These insects feed from inside grain kernels, making infestation invisible until weeks of damage have occurred.

By the time a farmer detects infestation through smell or visible insects, 10 to 20 percent of stored grain may already be lost. Existing detection methods require laboratory equipment, trained specialists, or commercial acoustic sensors costing hundreds of thousands of rupees. No affordable, offline, phone-based early warning tool existed for the 100 million small and marginal farmers who store grain at home or in kachha storage.

### Target Audience (Metric 1: Target Audience)
Small and marginal farmers in rural India storing rice, wheat, or other cereals at home, in jute sacks, or in kachha storage facilities. Krishi Vigyan Kendras, Farmer Producer Organisations, and ASHA-equivalent agricultural extension workers who conduct farm visits. Mobile penetration in rural India exceeds 70 percent (ASER 2024), making a phone-based solution genuinely reachable.

### Solution
Farmers hold their phone against a grain storage bag or bin and record 30 seconds of audio. Kaan converts the audio to a mel spectrogram and runs a compact INT8-quantized CNN that classifies it as clean grain, rice weevil, lesser grain borer, or red flour beetle. Results appear in seconds with the pest class, confidence score, and a specific actionable advisory in the farmer's chosen language.

### Why AI is Necessary (Metric 2: AI Not a Force-Fit)
These three pest species produce overlapping acoustic signatures in the 300 to 4000 Hz frequency range. Rice weevil produces signals concentrated around 355 to 371 Hz. Lesser grain borer and red flour beetle produce distinct but partially overlapping patterns. Rule-based thresholds based on energy or simple spectral features cannot reliably separate the three species. A trained CNN operating on mel spectrograms learns the subtle multi-frequency patterns that distinguish species-level acoustic signatures, a task that is genuinely impossible without machine learning.

### Model Performance (Metric 2: AI Knowledge and Execution)
Training dataset: IRRI Rice Acoustic Sensor Dataset (Balingbing et al., Computers and Electronics in Agriculture 225, 2024). 500 WAV recordings per class across four classes: clean, rice weevil, lesser grain borer, red flour beetle. 2,000 total training samples. Patient-level leakage-aware splits used.

Validation accuracy: 97.76 percent

Macro F1: 0.98

This exceeds the reference paper's reported accuracy of 84.51 percent on the same dataset.

Model size: about 333 KB (INT8 quantized TFLite), runs on-device

Architecture: mel spectrogram CNN, INT8 quantization, Streamlit + LiteRT / TensorFlow Lite inference (same model family as the public web app)

### Novel Contribution (Metric 2: Idea Novelty)
Every existing acoustic grain pest detection system uses dedicated hardware: Raspberry Pi Zero with Adafruit SPH0645 MEMS microphone (Balingbing et al. 2024), USDA piezoelectric sensors, commercial acoustic probes costing Rs 2 lakh or more. Kaan is the first system to run acoustic grain pest detection on a standard smartphone microphone, offline, in Indian languages, at no cost. The underlying acoustic technique is established in the literature. The delivery mechanism for Indian smallholder farmers is the novel contribution. Additionally, Kaan provides severity estimation from acoustic signal density (RMS energy and impulse rate), a feature absent from any published acoustic pest detection system.

### Technical Stack (Metric 3: Technical Knowledge)
Audio processing: librosa, mel spectrogram (128 bands, 16kHz, n_fft 2048, hop_length 512)

Model: custom CNN trained in TensorFlow/Keras

Quantization: INT8 via TensorFlow Lite (`project-kaan.tflite`)

Frontend (this repo): Streamlit prototype with multilingual advisories

Public web app: Next.js + ONNX Runtime Web at https://kaan-web.vercel.app (same trained model)

Languages: English, Hindi, Marathi, Punjabi, Telugu

Deployment: Streamlit Community Cloud / local; web on Vercel (zero backend cost for classification)

Hardware target: any Android or iOS smartphone with built-in microphone. INT8 model designed for AI-optimised mobile chipsets including Qualcomm Hexagon NPU for accelerated on-device inference.

Model size before quantization: about 1.3 MB (Keras H5)

Model size after INT8 quantization: about 333 KB (TFLite)

Compression ratio: about 4x relative to the Keras H5 checkpoint used in this release

### Accessibility and Inclusion (Metric 1: Inclusion Lines)
- Five Indian languages covering over 700 million native speakers
- Results communicated via symbol plus text, never colour alone, for colour-blind accessibility
- Guided tour on the public web app for first-time users who have never used an AI tool
- Works on any basic smartphone with a built-in microphone
- Fully offline: no internet connection required for inference
- No registration, login, or personal data collection required
- Designed for low-literacy use: colour-coded alerts with accompanying text labels

### Environmental Impact (Metric 1 and 2: Environmental Lines)
Early detection reduces prophylactic chemical fumigation, cutting pesticide use and grain contamination. The INT8 model requires minimal compute and produces negligible carbon output compared to cloud-based AI alternatives. Protecting stored grain directly reduces post-harvest food waste, contributing to climate-resilient food systems.

### Privacy and Ethics (Metric 2: Ethics and Privacy)
Audio is processed locally for classification (TFLite / LiteRT in this app; ONNX Runtime Web in the public site). No recordings are uploaded, stored, or transmitted to any server as part of inference. Kaan is explicitly framed as a screening aid, not a diagnostic instrument. Every result includes a disclaimer that professional inspection is recommended. The model reports confidence and flags low-confidence results with a request to re-record rather than delivering a false confident answer. Rice weevil showed the lowest F1 score consistent with its complex acoustic signature, and this limitation is documented. Pulse beetle and legume storage pests are not covered and are listed as future work.

### SDG Mapping (Metric 1: SDG Lines)
SDG 2 (Zero Hunger): Reduces post-harvest grain loss for smallholder farmers, directly supporting food security.

SDG 12 (Responsible Consumption and Production): Reduces prophylactic pesticide use through targeted, evidence-based intervention.

### GTM and Sustainability (Metric 1: Sustainable Pathways)
Distribution pathway: Krishi Vigyan Kendras (731 district-level agricultural extension centres across India), Farmer Producer Organisations, and state agriculture departments. One URL shared by one KVK officer per district reaches thousands of farmers without requiring app store deployment. Released under MIT licence. The open retraining pipeline allows state agriculture departments to retrain the model on locally recorded audio from specific regions, grain varieties, and storage conditions, enabling long-term sustainability without dependence on the original developers.

### Responsible AI Principles
Kaan follows the Promote Equity and Inclusion principle by providing a free, offline, multilingual tool that reaches populations excluded from conventional agricultural technology. It follows Enable Transparency and Explainability by showing confidence scores, spectrogram visualizations, and explicit uncertainty flags. It follows Advance Security, Safety, and Reliability by framing all outputs as screening recommendations requiring professional confirmation.

### Dataset Citation
Balingbing C et al. (2024). Application of a multi-layer CNN to classify major insect pests in stored rice detected by an acoustic device. Computers and Electronics in Agriculture, 225, 109297.

IGMRI (2015). Annual Report. Indian Grain Storage Management and Research Institute, Ministry of Food and Public Distribution, Government of India.

### Live Demo
https://kaan-web.vercel.app

Web source: https://github.com/arnavd371/kaan-web

### Licence
MIT

### Run locally (this repo)
```bash
pip install -r requirements.txt
streamlit run app.py
```

Training scripts live under `model/`. Retrain and export TFLite with `model/train.py` / `model/train_kaggle.py` and `model/convert_tflite.py`.

Copyright (c) 2026 Arnav Dhiman.
