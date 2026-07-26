"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import GuideTour from "@/components/GuideTour";
import SummitSummary from "@/components/SummitSummary";
import { LANGUAGES, type Lang } from "@/lib/language";
import {
  INTRO_GUIDE_ORDER,
  getGuideChrome,
  writeGuideEnabled,
} from "@/lib/guideCopy";

const MANIFESTO: Record<Lang, string> = {
  en: "Somewhere in rural India, a farmer is checking his grain by hand right now. He presses his palm into the sack, feels for heat, smells for something off. This is how it has been done for generations. It works, until it doesn't - until the infestation is already 10-20% deep before anyone notices. Kaan listens instead. It hears what you cannot.",
  hi: "भारत के किसी गांव में, एक किसान अभी अपने हाथ से अनाज की जांच कर रहा है। वह बोरी में हाथ दबाता है, गर्मी महसूस करता है, गंध सूंघता है। यह पीढ़ियों से इसी तरह किया जाता रहा है। यह तब तक काम करता है जब तक संक्रमण पहले से ही 10-20% गहरा नहीं हो जाता। Kaan इसके बजाय सुनता है। यह वह सुनता है जो आप नहीं सुन सकते।",
  mr: "भारतातील एका गावात, एक शेतकरी आत्ता हाताने धान्य तपासत आहे. तो पोत्यात हात दाबतो, उष्णता जाणवतो, वास घेतो. पिढ्यानपिढ्या असेच केले जात आहे. जोपर्यंत संसर्ग आधीच 10-20% खोल होत नाही तोपर्यंत हे चालते. Kaan त्याऐवजी ऐकतो. जे तुम्ही ऐकू शकत नाही ते तो ऐकतो.",
  pa: "ਭਾਰਤ ਦੇ ਕਿਸੇ ਪਿੰਡ ਵਿੱਚ, ਇੱਕ ਕਿਸਾਨ ਹੁਣੇ ਆਪਣੇ ਹੱਥ ਨਾਲ ਅਨਾਜ ਦੀ ਜਾਂਚ ਕਰ ਰਿਹਾ ਹੈ। ਉਹ ਬੋਰੀ ਵਿੱਚ ਹੱਥ ਦਬਾਉਂਦਾ ਹੈ, ਗਰਮੀ ਮਹਿਸੂਸ ਕਰਦਾ ਹੈ, ਗੰਧ ਸੁੰਘਦਾ ਹੈ। ਇਹ ਪੀੜ੍ਹੀਆਂ ਤੋਂ ਇਸੇ ਤਰ੍ਹਾਂ ਕੀਤਾ ਜਾਂਦਾ ਰਿਹਾ ਹੈ। ਇਹ ਉਦੋਂ ਤੱਕ ਕੰਮ ਕਰਦਾ ਹੈ ਜਦੋਂ ਤੱਕ ਲਗਾਤਾਰ ਪਹਿਲਾਂ ਹੀ 10-20% ਡੂੰਘੀ ਨਹੀਂ ਹੋ ਜਾਂਦੀ। Kaan ਇਸਦੀ ਬਜਾਏ ਸੁਣਦਾ ਹੈ। ਇਹ ਉਹ ਸੁਣਦਾ ਹੈ ਜੋ ਤੁਸੀਂ ਨਹੀਂ ਸੁਣ ਸਕਦੇ।",
  te: "భారతదేశంలోని ఒక గ్రామంలో, ఒక రైతు ఇప్పుడే తన చేతితో ధాన్యాన్ని పరిశీలిస్తున్నాడు. అతను సంచిలో అరచేతిని నొక్కుతాడు, వేడిని అనుభూతి చెందుతాడు, వాసన చూస్తాడు. ఇది తరతరాలుగా ఇలాగే జరుగుతోంది. సంక్రమణ ఇప్పటికే 10-20% లోతుగా ఉన్నప్పుడు మాత్రమే ఎవరైనా గమనించే వరకు ఇది పనిచేస్తుంది. Kaan బదులుగా వింటుంది. మీరు వినలేనిది అది వింటుంది.",
};

export default function IntroPage() {
  const router = useRouter();
  const [lang, setLang] = useState<Lang>("en");
  const [guideOn, setGuideOn] = useState(true);
  const [tourActive, setTourActive] = useState(false);
  const chrome = getGuideChrome(lang);

  useEffect(() => {
    const key = "kaan-guide-enabled";
    const raw = window.localStorage.getItem(key);
    const firstVisit = raw === null;
    const enabled = firstVisit ? true : raw === "1";
    setGuideOn(enabled);
    writeGuideEnabled(enabled);
    if (firstVisit && enabled) setTourActive(true);
  }, []);

  function setGuide(on: boolean) {
    setGuideOn(on);
    writeGuideEnabled(on);
  }

  function startApp() {
    router.push(guideOn ? "/app?guide=1" : "/app");
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-5 sm:px-6 py-12 sm:py-16 text-center">
      <div className="max-w-2xl w-full">
        <h1 className="text-5xl font-bold mb-1">Kaan</h1>
        <p className="text-2xl mb-8">कान</p>
        <p className="font-mono text-sm uppercase tracking-wide mb-10">It hears what you cannot.</p>

        <div data-guide="guide-intro-lang" className="flex justify-center gap-2 mb-10 flex-wrap">
          {(Object.keys(LANGUAGES) as Lang[]).map((code) => (
            <button
              key={code}
              type="button"
              onClick={() => setLang(code)}
              className={`btn-pi ${lang === code ? "!bg-black !text-white" : "btn-pi-ghost"}`}
            >
              {LANGUAGES[code]}
            </button>
          ))}
        </div>

        <p data-guide="guide-intro-story" className="text-lg leading-relaxed mb-10 text-left sm:text-center">
          {MANIFESTO[lang]}
        </p>

        <div data-guide="guide-intro-toggle" className="guide-toggle">
          <p className="font-mono text-xs uppercase tracking-wide">{chrome.toggle_label}</p>
          <div className="guide-toggle-row" role="group" aria-label={chrome.toggle_label}>
            <button type="button" aria-pressed={guideOn} onClick={() => setGuide(true)}>
              {chrome.toggle_on}
            </button>
            <button type="button" aria-pressed={!guideOn} onClick={() => setGuide(false)}>
              {chrome.toggle_off}
            </button>
          </div>
          <p className="guide-toggle-help">{chrome.toggle_help}</p>
          {!tourActive && (
            <button type="button" className="btn-pi btn-pi-ghost" onClick={() => setTourActive(true)}>
              {chrome.restart}
            </button>
          )}
        </div>

        <button
          data-guide="guide-intro-start"
          type="button"
          onClick={startApp}
          className="btn-pi w-full sm:w-auto px-10"
        >
          Listen to your grain
        </button>

        <SummitSummary />
      </div>

      <GuideTour
        lang={lang}
        steps={INTRO_GUIDE_ORDER}
        active={tourActive}
        onClose={() => setTourActive(false)}
      />
    </main>
  );
}
