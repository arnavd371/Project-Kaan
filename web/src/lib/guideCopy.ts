import type { Lang } from "./language";
import guide from "./guide.json";

type Chrome = {
  toggle_label: string;
  toggle_on: string;
  toggle_off: string;
  toggle_help: string;
  next: string;
  back: string;
  skip: string;
  done: string;
  step_of: string;
  restart: string;
};

type StepCopy = { title: string; body: string };

export type GuideStepId =
  | "intro_lang"
  | "intro_story"
  | "intro_toggle"
  | "intro_start"
  | "app_lang"
  | "app_tabs"
  | "app_upload"
  | "app_samples"
  | "app_how"
  | "app_about";

export const INTRO_GUIDE_ORDER: GuideStepId[] = [
  "intro_lang",
  "intro_story",
  "intro_toggle",
  "intro_start",
];

export const APP_GUIDE_ORDER: GuideStepId[] = [
  "app_lang",
  "app_tabs",
  "app_upload",
  "app_samples",
  "app_how",
  "app_about",
];

export const GUIDE_TARGETS: Record<GuideStepId, string> = {
  intro_lang: "guide-intro-lang",
  intro_story: "guide-intro-story",
  intro_toggle: "guide-intro-toggle",
  intro_start: "guide-intro-start",
  app_lang: "guide-app-lang",
  app_tabs: "guide-app-tabs",
  app_upload: "guide-app-upload",
  app_samples: "guide-app-samples",
  app_how: "guide-app-tabs",
  app_about: "guide-app-tabs",
};

export const GUIDE_TAB: Partial<Record<GuideStepId, "detect" | "how" | "about">> = {
  app_upload: "detect",
  app_samples: "detect",
  app_how: "how",
  app_about: "about",
};

const CHROME = guide.chrome as Record<Lang, Chrome>;
const STEPS = guide.steps as Record<Lang, Record<GuideStepId, StepCopy>>;

export function getGuideChrome(lang: Lang): Chrome {
  return CHROME[lang] ?? CHROME.en;
}

export function getGuideStep(lang: Lang, id: GuideStepId): StepCopy {
  return STEPS[lang]?.[id] ?? STEPS.en[id];
}

export function guideEnabledKey() {
  return "kaan-guide-enabled";
}

export function readGuideEnabled(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(guideEnabledKey()) === "1";
}

export function writeGuideEnabled(on: boolean) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(guideEnabledKey(), on ? "1" : "0");
}
