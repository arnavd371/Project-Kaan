import type { DemoPrediction } from "./audio";
import { loadMelFilterbank, preprocessWaveform, MEL_SHAPE } from "./mel";

export const CLASS_NAMES = ["clean", "rice_weevil", "lesser_grain_borer", "red_flour_beetle"] as const;

const CONFIDENCE_THRESHOLD = 0.6;
const INPUT_SCALE = 0.003921568859368563;
const INPUT_ZERO_POINT = 0;
const OUTPUT_SCALE = 0.00390625;
const OUTPUT_ZERO_POINT = 0;
const INPUT_NAME = "serving_default_input_layer_2:0";
const ORT_VERSION = "1.27.0";
const ORT_SCRIPT = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.js`;
const ORT_WASM_PATH = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

type OrtTensor = { data: ArrayLike<number> };
type OrtSession = {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor>>;
};
type OrtNamespace = {
  env: { wasm: { wasmPaths: string; numThreads: number; simd: boolean } };
  Tensor: new (type: string, data: Uint8Array, dims: number[]) => unknown;
  InferenceSession: {
    create: (url: string, opts: Record<string, unknown>) => Promise<OrtSession>;
  };
};

let ortPromise: Promise<OrtNamespace> | null = null;
let sessionPromise: Promise<OrtSession> | null = null;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[data-kaan-ort="1"]`);
    if (existing) {
      if ((window as unknown as { ort?: OrtNamespace }).ort) {
        resolve();
        return;
      }
      existing.addEventListener("load", () => resolve());
      existing.addEventListener("error", () => reject(new Error("Failed to load ONNX Runtime")));
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.dataset.kaanOrt = "1";
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

async function getOrt(): Promise<OrtNamespace> {
  if (!ortPromise) {
    ortPromise = (async () => {
      await loadScript(ORT_SCRIPT);
      const ort = (window as unknown as { ort?: OrtNamespace }).ort;
      if (!ort) throw new Error("ONNX Runtime failed to initialize");
      ort.env.wasm.wasmPaths = ORT_WASM_PATH;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      return ort;
    })();
  }
  return ortPromise;
}

function quantizeInput(mel: Float32Array): Uint8Array {
  const out = new Uint8Array(mel.length);
  for (let i = 0; i < mel.length; i++) {
    const q = Math.round(mel[i] / INPUT_SCALE + INPUT_ZERO_POINT);
    out[i] = Math.max(0, Math.min(255, q));
  }
  return out;
}

function toPrediction(raw: ArrayLike<number>): DemoPrediction {
  const probs = new Float32Array(raw.length);
  let sum = 0;
  for (let i = 0; i < raw.length; i++) {
    probs[i] = Math.max(0, (Number(raw[i]) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE);
    sum += probs[i];
  }
  if (sum > 0) {
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  }

  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;

  const allScores: Record<string, number> = {};
  for (let i = 0; i < CLASS_NAMES.length; i++) allScores[CLASS_NAMES[i]] = probs[i];

  return {
    predictedClass: CLASS_NAMES[best],
    confidence: probs[best],
    confident: probs[best] > CONFIDENCE_THRESHOLD,
    allScores,
  };
}

export async function loadKaanModel(): Promise<OrtSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      await loadMelFilterbank("/model/mel_fb.f32");
      const ort = await getOrt();
      return ort.InferenceSession.create("/model/project-kaan.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })();
  }
  return sessionPromise;
}

export async function predictCnn(samples: Float32Array): Promise<DemoPrediction> {
  const ort = await getOrt();
  const session = await loadKaanModel();
  const mel = preprocessWaveform(samples);
  const quantized = quantizeInput(mel);
  const input = new ort.Tensor("uint8", quantized, [1, MEL_SHAPE, MEL_SHAPE, 1]);
  const results = await session.run({ [INPUT_NAME]: input });
  const output = Object.values(results)[0];
  return toPrediction(output.data);
}

export function prefetchModel(): void {
  void loadKaanModel().catch(() => {});
}
