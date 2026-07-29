export const SAMPLE_RATE = 16000;
export const DURATION_SEC = 10;
export const N_SAMPLES = SAMPLE_RATE * DURATION_SEC;
export const N_MELS = 128;
export const N_FFT = 2048;
export const HOP_LENGTH = 512;
export const MEL_SHAPE = 128;
export const N_FREQS = N_FFT / 2 + 1; // 1025

let melFb: Float32Array | null = null; // row-major [128, 1025]

export async function loadMelFilterbank(url = "/model/mel_fb.f32"): Promise<void> {
  if (melFb) return;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load mel filterbank (${res.status})`);
  const buf = await res.arrayBuffer();
  const expected = N_MELS * N_FREQS * 4;
  if (buf.byteLength !== expected) {
    throw new Error(`mel_fb.f32 size ${buf.byteLength}, expected ${expected}`);
  }
  melFb = new Float32Array(buf);
}

export function setMelFilterbank(data: Float32Array): void {
  if (data.length !== N_MELS * N_FREQS) {
    throw new Error(`mel filterbank length ${data.length}, expected ${N_MELS * N_FREQS}`);
  }
  melFb = data;
}

export function trimSilence(y: Float32Array, topDb = 20, frameLength = 2048, hopLength = 512): Float32Array {
  if (y.length === 0) return y.slice();
  const pad = Math.floor(frameLength / 2);
  const padded = new Float32Array(y.length + 2 * pad);
  padded.set(y, pad);
  const nFrames = 1 + Math.floor(y.length / hopLength);
  const rms = new Float64Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    const start = i * hopLength;
    let sum = 0;
    for (let j = 0; j < frameLength; j++) {
      const v = padded[start + j];
      sum += v * v;
    }
    rms[i] = Math.sqrt(sum / frameLength);
  }
  let ref = 0;
  for (let i = 0; i < nFrames; i++) if (rms[i] > ref) ref = rms[i];
  if (ref < 1e-12) return y.slice();
  const thresh = ref * Math.pow(10, -topDb / 20);
  let first = 0;
  while (first < nFrames && rms[first] < thresh) first++;
  let last = nFrames - 1;
  while (last >= 0 && rms[last] < thresh) last--;
  if (first > last) return y.slice();
  // librosa.effects.trim: frames_to_samples(first) .. frames_to_samples(last + 1)
  const start = first * hopLength;
  const end = Math.min(y.length, (last + 1) * hopLength);
  return y.subarray(start, end);
}

export function trimAndPad(y: Float32Array): Float32Array {
  let yt = trimSilence(y);
  if (yt.length === 0) yt = y;
  if (yt.length >= N_SAMPLES) {
    const start = Math.floor((yt.length - N_SAMPLES) / 2);
    return Float32Array.from(yt.subarray(start, start + N_SAMPLES));
  }
  const out = new Float32Array(N_SAMPLES);
  const padLeft = Math.floor((N_SAMPLES - yt.length) / 2);
  out.set(yt, padLeft);
  return out;
}

function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
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
}

function hannWindow(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  }
  return w;
}

function zoomLinear(
  src: Float64Array,
  srcH: number,
  srcW: number,
  outH: number,
  outW: number
): Float32Array {
  const out = new Float32Array(outH * outW);
  const scaleY = outH === 1 ? 0 : (srcH - 1) / (outH - 1);
  const scaleX = outW === 1 ? 0 : (srcW - 1) / (outW - 1);
  for (let y = 0; y < outH; y++) {
    const sy = y * scaleY;
    const y0 = Math.floor(sy);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const wy = sy - y0;
    const y0c = Math.max(0, Math.min(srcH - 1, y0));
    for (let x = 0; x < outW; x++) {
      const sx = x * scaleX;
      const x0 = Math.floor(sx);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const wx = sx - x0;
      const x0c = Math.max(0, Math.min(srcW - 1, x0));
      const v00 = src[y0c * srcW + x0c];
      const v01 = src[y0c * srcW + x1];
      const v10 = src[y1 * srcW + x0c];
      const v11 = src[y1 * srcW + x1];
      out[y * outW + x] = v00 * (1 - wy) * (1 - wx) + v01 * (1 - wy) * wx + v10 * wy * (1 - wx) + v11 * wy * wx;
    }
  }
  return out;
}

export function waveformToMelSpectrogram(y: Float32Array): Float32Array {
  if (!melFb) {
    throw new Error("Mel filterbank not loaded - call loadMelFilterbank() first");
  }

  const pad = N_FFT / 2;
  const padded = new Float32Array(y.length + 2 * pad);
  padded.set(y, pad); // pad_mode='constant'

  // librosa uses periodic Hann for STFT (get_window('hann', nx, fftbins=True))
  const window = hannWindow(N_FFT);
  const nFrames = 1 + Math.floor(y.length / HOP_LENGTH);
  const power = new Float64Array(N_FREQS * nFrames);

  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);

  for (let f = 0; f < nFrames; f++) {
    const start = f * HOP_LENGTH;
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < N_FFT; i++) {
      re[i] = padded[start + i] * window[i];
    }
    fft(re, im);
    for (let k = 0; k < N_FREQS; k++) {
      // librosa STFT scaling: usually 1.0 for power spectrogram path via abs**2
      power[k * nFrames + f] = re[k] * re[k] + im[k] * im[k];
    }
  }

  const mel = new Float64Array(N_MELS * nFrames);
  for (let m = 0; m < N_MELS; m++) {
    const base = m * N_FREQS;
    for (let f = 0; f < nFrames; f++) {
      let s = 0;
      for (let k = 0; k < N_FREQS; k++) {
        const w = melFb[base + k];
        if (w !== 0) s += w * power[k * nFrames + f];
      }
      mel[m * nFrames + f] = s;
    }
  }

  // power_to_db(ref=np.max), amin=1e-10, top_db=80
  let ref = 1e-10;
  for (let i = 0; i < mel.length; i++) if (mel[i] > ref) ref = mel[i];
  const amin = 1e-10;
  for (let i = 0; i < mel.length; i++) {
    mel[i] = 10 * Math.log10(Math.max(amin, mel[i]) / Math.max(amin, ref));
  }
  let dbMax = -Infinity;
  for (let i = 0; i < mel.length; i++) if (mel[i] > dbMax) dbMax = mel[i];
  const floor = dbMax - 80;
  for (let i = 0; i < mel.length; i++) if (mel[i] < floor) mel[i] = floor;

  let dbMin = Infinity;
  dbMax = -Infinity;
  for (let i = 0; i < mel.length; i++) {
    if (mel[i] < dbMin) dbMin = mel[i];
    if (mel[i] > dbMax) dbMax = mel[i];
  }
  const span = dbMax - dbMin;
  if (span > 1e-8) {
    for (let i = 0; i < mel.length; i++) mel[i] = (mel[i] - dbMin) / span;
  } else {
    mel.fill(0);
  }

  return zoomLinear(mel, N_MELS, nFrames, MEL_SHAPE, MEL_SHAPE);
}

export function preprocessWaveform(y: Float32Array): Float32Array {
  return waveformToMelSpectrogram(trimAndPad(y));
}
