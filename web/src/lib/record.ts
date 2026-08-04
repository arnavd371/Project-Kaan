/** In-app contact recording for phone-on-bag screening. */

const TARGET_MS = 10_000;

export type RecordProgress = {
  elapsedMs: number;
  targetMs: number;
  recording: boolean;
};

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t));
}

export async function recordContactClip(
  onProgress?: (p: RecordProgress) => void,
  targetMs = TARGET_MS
): Promise<File> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone is not available on this device.");
  }
  if (typeof MediaRecorder === "undefined") {
    throw new Error("Recording is not supported in this WebView.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });

  const mimeType = pickMimeType();
  const recorder = mimeType
    ? new MediaRecorder(stream, { mimeType })
    : new MediaRecorder(stream);
  const chunks: BlobPart[] = [];

  recorder.ondataavailable = (ev) => {
    if (ev.data.size > 0) chunks.push(ev.data);
  };

  const started = Date.now();
  onProgress?.({ elapsedMs: 0, targetMs, recording: true });
  const tick = window.setInterval(() => {
    onProgress?.({
      elapsedMs: Math.min(Date.now() - started, targetMs),
      targetMs,
      recording: true,
    });
  }, 200);

  try {
    await new Promise<void>((resolve, reject) => {
      recorder.onerror = () => reject(new Error("Recording failed."));
      recorder.onstop = () => resolve();
      recorder.start(250);
      window.setTimeout(() => {
        if (recorder.state !== "inactive") recorder.stop();
      }, targetMs);
    });
  } finally {
    window.clearInterval(tick);
    stream.getTracks().forEach((t) => t.stop());
    onProgress?.({ elapsedMs: targetMs, targetMs, recording: false });
  }

  const type = recorder.mimeType || mimeType || "audio/webm";
  const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
  const blob = new Blob(chunks, { type });
  if (blob.size < 1000) {
    throw new Error("Recording was too short or empty. Try again in a quieter place.");
  }
  return new File([blob], `kaan-contact.${ext}`, { type });
}
