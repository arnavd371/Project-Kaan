"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import GuideTour from "@/components/GuideTour";
import { LANGUAGES, CLASS_NAMES, type Lang, getUi, getAdvisory, getPestName } from "@/lib/language";
import { decodeToMono16k, estimateSeverity, type DemoPrediction, type SeverityResult } from "@/lib/audio";
import { predictCnn, prefetchModel } from "@/lib/model";
import {
  APP_GUIDE_ORDER,
  GUIDE_TAB,
  getGuideChrome,
  readGuideEnabled,
  writeGuideEnabled,
  type GuideStepId,
} from "@/lib/guideCopy";

type Tab = "detect" | "how" | "about";

const IRRI_SAMPLES: { id: string; file: string; labelKey: string }[] = [
  { id: "rice_weevil", file: "/samples/rice_weevil.wav", labelKey: "sample_rice_weevil" },
  { id: "lesser_grain_borer", file: "/samples/lesser_grain_borer.wav", labelKey: "sample_lgb" },
  { id: "red_flour_beetle", file: "/samples/red_flour_beetle.wav", labelKey: "sample_rfb" },
  { id: "clean", file: "/samples/clean.wav", labelKey: "sample_clean" },
];

const ABOUT_SECTIONS = [
  "about_problem",
  "about_audience",
  "about_sdg",
  "about_env",
  "about_privacy",
  "about_limitations",
  "about_gtm",
  "about_why_ai",
  "about_inclusion",
  "about_ethics",
  "about_innovation",
  "about_techstack",
  "about_deployment",
] as const;

const SEVERITY_LABELS: Record<Lang, Record<"Early" | "Moderate" | "Severe", string>> = {
  en: { Early: "Early", Moderate: "Moderate", Severe: "Severe" },
  hi: { Early: "प्रारंभिक", Moderate: "मध्यम", Severe: "गंभीर" },
  mr: { Early: "सुरुवातीचा", Moderate: "मध्यम", Severe: "गंभीर" },
  pa: { Early: "ਸ਼ੁਰੂਆਤੀ", Moderate: "ਦਰਮਿਆਨਾ", Severe: "ਗੰਭੀਰ" },
  te: { Early: "ప్రారంభ", Moderate: "మధ్యస్థ", Severe: "తీవ్రమైన" },
};

async function loadDemoSample(): Promise<File> {
  const res = await fetch("/samples/rice_weevil.wav");
  if (!res.ok) throw new Error("Could not load demo sample.");
  const blob = await res.blob();
  return new File([blob], "rice_weevil.wav", { type: "audio/wav" });
}

export default function AppPage() {
  const [lang, setLang] = useState<Lang>("en");
  const [tab, setTab] = useState<Tab>("detect");
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<DemoPrediction | null>(null);
  const [severity, setSeverity] = useState<SeverityResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tourActive, setTourActive] = useState(false);
  const chrome = getGuideChrome(lang);

  useEffect(() => {
    prefetchModel();
  }, []);

  useEffect(() => {
    const fromQuery = new URLSearchParams(window.location.search).get("guide") === "1";
    const fromStorage = readGuideEnabled();
    if (fromQuery || fromStorage) {
      setTourActive(true);
    }
  }, []);

  const onGuideStep = useCallback((stepId: GuideStepId) => {
    const nextTab = GUIDE_TAB[stepId];
    if (nextTab) setTab(nextTab);
  }, []);

  useEffect(() => {
    if (!file) {
      setAudioUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function analyze(targetFile: File) {
    setAnalyzing(true);
    setError(null);
    setResult(null);
    setSeverity(null);
    try {
      const samples = await decodeToMono16k(targetFile);
      const prediction = await predictCnn(samples);
      setResult(prediction);
      if (prediction.predictedClass !== "clean") {
        setSeverity(estimateSeverity(samples));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not analyze this audio file.");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <main className="min-h-screen px-5 sm:px-6 py-8 sm:py-10 max-w-3xl mx-auto">
      <header className="flex items-center justify-between mb-6 sm:mb-8 flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold">Kaan</h1>
          <p className="text-sm">{getUi(lang, "app_tagline")}</p>
        </div>
        <div className="flex items-center gap-2 sm:gap-3">
          <div data-guide="guide-app-lang">
          <select
            value={lang}
            onChange={(e) => setLang(e.target.value as Lang)}
            className="btn-pi"
            aria-label={getUi(lang, "language_label")}
          >
            {(Object.keys(LANGUAGES) as Lang[]).map((code) => (
              <option key={code} value={code}>
                {LANGUAGES[code]}
              </option>
            ))}
          </select>
          </div>
          <Link href="/" className="btn-pi btn-pi-ghost inline-flex items-center">
            Back
          </Link>
          <button
            type="button"
            className="btn-pi btn-pi-ghost"
            onClick={() => {
              writeGuideEnabled(true);
              setTourActive(true);
            }}
          >
            {chrome.restart}
          </button>
        </div>
      </header>

      <nav data-guide="guide-app-tabs" className="kaan-tabs" aria-label="Sections">
        {(["detect", "how", "about"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            aria-current={tab === t ? "page" : undefined}
          >
            {t === "detect" ? getUi(lang, "tab_detect") : t === "how" ? getUi(lang, "tab_how") : getUi(lang, "tab_about")}
          </button>
        ))}
      </nav>

      {tab === "detect" && (
        <section>
          <label className="block font-mono text-xs uppercase mb-2">{getUi(lang, "upload_label")}</label>
          <p className="text-sm mb-4 text-black/70">{getUi(lang, "upload_help")}</p>
          <div data-guide="guide-app-upload" className="flex flex-col sm:flex-row flex-wrap gap-3 mb-6 items-stretch sm:items-center">
            <label className="btn-pi cursor-pointer inline-flex items-center justify-center">
              <span>Choose audio file</span>
              <input
                type="file"
                accept="audio/*,.wav,.mp3,.m4a,.ogg"
                onChange={(e) => {
                  const f = e.target.files?.[0] ?? null;
                  setFile(f);
                  if (f) analyze(f);
                }}
                className="sr-only"
              />
            </label>
            <button
              type="button"
              disabled={analyzing}
              onClick={async () => {
                try {
                  const demoFile = await loadDemoSample();
                  setFile(demoFile);
                  await analyze(demoFile);
                } catch (e) {
                  setError(e instanceof Error ? e.message : "Could not load demo sample.");
                }
              }}
              className="btn-pi"
            >
              {getUi(lang, "try_demo_btn")}
            </button>
            {file && (
              <button
                type="button"
                onClick={() => {
                  setFile(null);
                  setResult(null);
                  setSeverity(null);
                  setError(null);
                }}
                className="btn-pi btn-pi-ghost"
              >
                {getUi(lang, "clear_btn")}
              </button>
            )}
          </div>

          <div data-guide="guide-app-samples" className="panel mb-6">
            <p className="panel-label mb-1">{getUi(lang, "sample_section_title")}</p>
            <p className="text-sm text-black/70 mb-3">{getUi(lang, "sample_section_help")}</p>
            <div className="sample-grid mb-3">
              {IRRI_SAMPLES.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  disabled={analyzing}
                  onClick={async () => {
                    const res = await fetch(sample.file);
                    const blob = await res.blob();
                    const sampleFile = new File([blob], `${sample.id}.wav`, { type: "audio/wav" });
                    setFile(sampleFile);
                    await analyze(sampleFile);
                  }}
                  className="btn-pi w-full"
                >
                  {getUi(lang, sample.labelKey)}
                </button>
              ))}
            </div>
            <p className="text-xs text-black/50">
              <a
                href="https://github.com/cbalingbing/Rice-Acoustic-Sensor"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                {getUi(lang, "sample_attribution")}
              </a>
            </p>
          </div>

          {audioUrl && <audio controls src={audioUrl} className="w-full mb-4" />}

          {analyzing && <p className="font-mono text-sm">{getUi(lang, "analyzing")}</p>}
          {error && <p className="text-red-700 text-sm">{error}</p>}

          {result && (
            <div className="panel">
              <p className="text-2xl font-bold mb-1">
                {result.predictedClass === "clean" ? "✅" : result.confident ? "⚠️" : "🔶"}{" "}
                {result.predictedClass === "clean" ? getUi(lang, "result_clean") : getUi(lang, "result_pest")}
              </p>
              {result.predictedClass !== "clean" && (
                <p className="font-semibold mb-2">{getPestName(lang, result.predictedClass)}</p>
              )}
              <p className="text-sm text-black/70 mb-1">
                {getUi(lang, "confidence_label")}: {(result.confidence * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-black/50 mb-4">{getUi(lang, "a11y_result_note")}</p>

              <p className="font-mono text-xs uppercase mb-1">{getUi(lang, "all_scores_title")}</p>
              <div className="space-y-1 mb-4">
                {CLASS_NAMES.map((cn) => (
                  <div key={cn} className="flex items-center gap-2 text-xs">
                    <span className="w-40 truncate">{getPestName(lang, cn)}</span>
                    <div className="flex-1 bg-black/10 h-2">
                      <div
                        className="bg-black h-2"
                        style={{ width: `${(result.allScores[cn] ?? 0) * 100}%` }}
                      />
                    </div>
                    <span className="w-10 text-right">{((result.allScores[cn] ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>

              <p className="font-mono text-xs uppercase mb-1">{getUi(lang, "advisory_title")}</p>
              <p className="text-sm mb-3">{getAdvisory(lang, result.predictedClass)}</p>
              <p className="text-xs text-black/50">{getUi(lang, "disclaimer")}</p>
            </div>
          )}

          {severity && (
            <div className="panel">
              <span className="panel-label">Severity</span>
              <div className="flex gap-2 my-3 font-mono text-xs uppercase">
                {(["Early", "Moderate", "Severe"] as const).map((lvl) => (
                  <span
                    key={lvl}
                    className={`px-3 py-1 border ${
                      severity.level === lvl ? "bg-black text-white border-black" : "border-black/30 text-black/50"
                    }`}
                  >
                    {SEVERITY_LABELS[lang][lvl]}
                  </span>
                ))}
              </div>
              <p className="text-2xl mb-1">
                {severity.symbol} {SEVERITY_LABELS[lang][severity.level]}
              </p>
              <p className="text-sm mb-1">{severity.message}</p>
              <p className="text-sm font-semibold mb-3">Action: {severity.action}</p>
              <p className="text-xs text-black/50">
                Severity estimated from acoustic signal density (RMS energy and impulse rate). Methodology based on
                Balingbing et al., Computers and Electronics in Agriculture, 2024.
              </p>
            </div>
          )}

          <p className="text-xs text-black/40 mt-6">{getUi(lang, "cnn_note")}</p>
        </section>
      )}

      {tab === "how" && (
        <section>
          <h2 className="text-2xl font-bold mb-4">{getUi(lang, "how_title")}</h2>
          {["how_step_1", "how_step_2", "how_step_3", "how_step_4"].map((k) => (
            <p key={k} className="mb-2 text-sm">
              {getUi(lang, k)}
            </p>
          ))}
          <div className="mt-6">
            {[
              ["how_mel_title", "how_mel_desc"],
              ["how_arch_title", "how_arch_desc"],
              ["how_datasets_title", "how_datasets_desc"],
              ["how_accuracy_title", "how_accuracy_desc"],
              ["how_freq_title", "how_freq_desc"],
            ].map(([titleKey, descKey], i) => (
              <div className="pi-timeline-item" key={titleKey}>
                <div className="pi-timeline-row">
                  <span className="pi-timeline-title">{getUi(lang, titleKey)}</span>
                  <span className="pi-timeline-meta">{String(i + 1).padStart(2, "0")}</span>
                </div>
                <p className="pi-timeline-body">{getUi(lang, descKey)}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {tab === "about" && (
        <section>
          <h2 className="text-2xl font-bold mb-4">{getUi(lang, "about_title")}</h2>
          <div>
            {ABOUT_SECTIONS.map((k, i) => (
              <div className="pi-timeline-item" key={k}>
                <div className="pi-timeline-row">
                  <span className="pi-timeline-title">{getUi(lang, `${k}_title`)}</span>
                  <span className="pi-timeline-meta">{String(i + 1).padStart(2, "0")}</span>
                </div>
                <p className="pi-timeline-body">{getUi(lang, k)}</p>
              </div>
            ))}
          </div>
          <div className="panel">
            <p className="font-bold mb-2">Team</p>
            <p>{getUi(lang, "about_team")}</p>
          </div>
          <div className="panel">
            <p className="font-bold mb-2">Open source & accessibility</p>
            <p className="mb-3">
              Kaan is released under the Apache License 2.0. The full training pipeline is on GitHub so state agriculture
              departments and Krishi Vigyan Kendras can retrain on locally recorded data for their region and grain
              variety.
            </p>
            <p className="mb-3">
              The INT8-quantized model is sized for on-device inference (including future packaging for Android NPU /
              APU paths) without requiring cloud compute for classification.
            </p>
            <p>
              Kaan never communicates a result through colour alone. Every result includes a symbol indicator so the
              app is usable by farmers with colour vision deficiency.
            </p>
          </div>
        </section>
      )}

      <GuideTour
        lang={lang}
        steps={APP_GUIDE_ORDER}
        active={tourActive}
        onClose={() => {
          setTourActive(false);
          writeGuideEnabled(false);
        }}
        onStepChange={onGuideStep}
      />
    </main>
  );
}
