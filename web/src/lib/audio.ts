const TARGET_SR = 16000;

export async function decodeToMono16k(file: File): Promise<Float32Array> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
  const tempCtx = new AudioCtx();
  const decoded = await tempCtx.decodeAudioData(arrayBuffer.slice(0));
  tempCtx.close();

  const channelCount = decoded.numberOfChannels;
  const length = decoded.length;
  const mono = new Float32Array(length);
  for (let c = 0; c < channelCount; c++) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < length; i++) mono[i] += data[i] / channelCount;
  }

  if (decoded.sampleRate === TARGET_SR) return mono;

  const offline = new OfflineAudioContext(
    1,
    Math.ceil((length * TARGET_SR) / decoded.sampleRate),
    TARGET_SR
  );
  const buffer = offline.createBuffer(1, length, decoded.sampleRate);
  buffer.copyToChannel(mono, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

export function rms(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

function frameRms(samples: Float32Array, frameLength = 2048, hop = 512): number[] {
  const frames: number[] = [];
  for (let start = 0; start + frameLength <= samples.length; start += hop) {
    let sum = 0;
    for (let i = start; i < start + frameLength; i++) sum += samples[i] * samples[i];
    frames.push(Math.sqrt(sum / frameLength));
  }
  return frames.length ? frames : [rms(samples)];
}

function fftMagnitudes(signal: Float32Array): Float64Array {
  let n = 1;
  while (n < signal.length) n *= 2;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < signal.length; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (signal.length - 1));
    re[i] = signal[i] * w;
  }
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
  const mags = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) mags[i] = Math.hypot(re[i], im[i]);
  return mags;
}

export function spectralCentroid(samples: Float32Array, sr = TARGET_SR): number {
  const frameLength = 2048;
  const hop = 512;
  const centroids: number[] = [];
  for (let start = 0; start + frameLength <= samples.length; start += hop) {
    const frame = samples.subarray(start, start + frameLength);
    const mags = fftMagnitudes(frame);
    let weighted = 0;
    let total = 0;
    for (let k = 0; k < mags.length; k++) {
      const freq = (k * sr) / (2 * mags.length);
      weighted += freq * mags[k];
      total += mags[k];
    }
    centroids.push(total > 0 ? weighted / total : 0);
  }
  if (centroids.length === 0) return 0;
  return centroids.reduce((a, b) => a + b, 0) / centroids.length;
}

export interface DemoPrediction {
  predictedClass: "clean" | "rice_weevil" | "lesser_grain_borer" | "red_flour_beetle";
  confidence: number;
  confident: boolean;
  allScores: Record<string, number>;
}

const DEMO_RMS_THRESHOLD = 0.02;
const DEMO_CENTROID_LOW = 300;
const DEMO_CENTROID_HIGH = 500;
const CONFIDENCE_THRESHOLD = 0.6;

export function demoPredict(samples: Float32Array): DemoPrediction {
  const energy = rms(samples);
  const centroid = spectralCentroid(samples);

  if (energy > DEMO_RMS_THRESHOLD && centroid >= DEMO_CENTROID_LOW && centroid <= DEMO_CENTROID_HIGH) {
    return {
      predictedClass: "rice_weevil",
      confidence: 0.72,
      confident: 0.72 > CONFIDENCE_THRESHOLD,
      allScores: { clean: 0.1, rice_weevil: 0.72, lesser_grain_borer: 0.1, red_flour_beetle: 0.08 },
    };
  }
  return {
    predictedClass: "clean",
    confidence: 0.85,
    confident: 0.85 > CONFIDENCE_THRESHOLD,
    allScores: { clean: 0.85, rice_weevil: 0.05, lesser_grain_borer: 0.05, red_flour_beetle: 0.05 },
  };
}

export interface SeverityResult {
  level: "Early" | "Moderate" | "Severe";
  color: string;
  symbol: string;
  message: string;
  action: string;
  urgency: 1 | 2 | 3;
}

export function estimateSeverity(samples: Float32Array, sr = TARGET_SR): SeverityResult {
  const frames = frameRms(samples);
  const mean = frames.reduce((a, b) => a + b, 0) / frames.length;
  const variance = frames.reduce((a, b) => a + (b - mean) ** 2, 0) / frames.length;
  const std = Math.sqrt(variance);
  const threshold = mean + 1.5 * std;
  const impulseFrames = frames.filter((f) => f > threshold).length;
  const durationSec = samples.length / sr;
  const impulseRate = impulseFrames / durationSec;

  if (mean < 0.015 || impulseRate < 2) {
    return {
      level: "Early",
      color: "#f59e0b",
      symbol: "🟡",
      message: "Early stage infestation. Insects present but population is low.",
      action: "Act within 2 weeks. Sun-dry grain and add neem leaves as preventive measure.",
      urgency: 1,
    };
  }
  if (mean < 0.04 || impulseRate < 8) {
    return {
      level: "Moderate",
      color: "#f97316",
      symbol: "🟠",
      message: "Moderate infestation. Active insect activity detected.",
      action: "Act within 3 days. Move grain to hermetic storage immediately.",
      urgency: 2,
    };
  }
  return {
    level: "Severe",
    color: "#ef4444",
    symbol: "🔴",
    message: "Severe infestation. Heavy insect activity detected.",
    action: "Act immediately. Inspect outer grain layer, discard heavily damaged portions, contact KVK today.",
    urgency: 3,
  };
}
