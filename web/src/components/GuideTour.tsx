"use client";

import { useEffect, useLayoutEffect, useState, type CSSProperties } from "react";
import type { Lang } from "@/lib/language";
import {
  GUIDE_TARGETS,
  getGuideChrome,
  getGuideStep,
  type GuideStepId,
} from "@/lib/guideCopy";

type Rect = { top: number; left: number; width: number; height: number };

type Props = {
  lang: Lang;
  steps: GuideStepId[];
  active: boolean;
  onClose: () => void;
  onStepChange?: (stepId: GuideStepId) => void;
};

function measureTarget(id: string): Rect | null {
  const el = document.querySelector(`[data-guide="${id}"]`) as HTMLElement | null;
  if (!el) return null;
  const r = el.getBoundingClientRect();
  if (r.width < 2 && r.height < 2) return null;
  const pad = 8;
  return {
    top: Math.max(8, r.top - pad),
    left: Math.max(8, r.left - pad),
    width: Math.min(window.innerWidth - 16, r.width + pad * 2),
    height: Math.min(window.innerHeight - 16, r.height + pad * 2),
  };
}

export default function GuideTour({ lang, steps, active, onClose, onStepChange }: Props) {
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<Rect | null>(null);
  const chrome = getGuideChrome(lang);
  const stepId = steps[index];
  const copy = stepId ? getGuideStep(lang, stepId) : null;
  const targetId = stepId ? GUIDE_TARGETS[stepId] : "";

  useEffect(() => {
    if (!active) {
      setIndex(0);
      setRect(null);
      return;
    }
    setIndex(0);
  }, [active, steps]);

  useEffect(() => {
    if (!active || !stepId) return;
    onStepChange?.(stepId);
  }, [active, stepId, onStepChange]);

  useLayoutEffect(() => {
    if (!active || !targetId) return;

    let cancelled = false;
    let el: HTMLElement | null = null;
    const update = () => {
      if (cancelled) return;
      document.querySelectorAll(".guide-target-active").forEach((node) => {
        node.classList.remove("guide-target-active");
      });
      el = document.querySelector(`[data-guide="${targetId}"]`) as HTMLElement | null;
      const next = measureTarget(targetId);
      setRect(next);
      if (el) {
        el.classList.add("guide-target-active");
        el.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    };

    const t = window.setTimeout(update, 80);
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
      document.querySelectorAll(".guide-target-active").forEach((node) => {
        node.classList.remove("guide-target-active");
      });
    };
  }, [active, targetId, index]);

  if (!active || !copy || !stepId) return null;

  const total = steps.length;
  const isLast = index >= total - 1;
  const stepLabel = chrome.step_of.replace("{n}", String(index + 1)).replace("{total}", String(total));

  const tipStyle: CSSProperties = (() => {
    if (!rect) {
      return { bottom: 24, left: "50%", transform: "translateX(-50%)", width: "min(22rem, calc(100vw - 2rem))" };
    }
    const spaceBelow = window.innerHeight - (rect.top + rect.height);
    const placeBelow = spaceBelow > 180 || rect.top < 160;
    const left = Math.min(Math.max(16, rect.left), window.innerWidth - 16 - Math.min(352, window.innerWidth - 32));
    if (placeBelow) {
      return {
        top: Math.min(window.innerHeight - 200, rect.top + rect.height + 12),
        left,
        width: "min(22rem, calc(100vw - 2rem))",
      };
    }
    return {
      bottom: Math.max(16, window.innerHeight - rect.top + 12),
      left,
      width: "min(22rem, calc(100vw - 2rem))",
    };
  })();

  return (
    <div className="guide-root" role="dialog" aria-modal="true" aria-labelledby="guide-title">
      {!rect && <div className="guide-dim guide-dim-visible" aria-hidden="true" />}
      {rect && (
        <div
          className="guide-spotlight"
          style={{
            top: rect.top,
            left: rect.left,
            width: rect.width,
            height: rect.height,
          }}
          aria-hidden="true"
        />
      )}
      {rect && <div className="guide-dim" aria-hidden="true" />}

      <div className="guide-card" style={tipStyle}>
        <p className="guide-step-meta">{stepLabel}</p>
        <h2 id="guide-title" className="guide-title">
          {copy.title}
        </h2>
        <p className="guide-body">{copy.body}</p>
        <div className="guide-actions">
          <button type="button" className="btn-pi btn-pi-ghost" onClick={onClose}>
            {chrome.skip}
          </button>
          <div className="guide-actions-right">
            {index > 0 && (
              <button type="button" className="btn-pi btn-pi-ghost" onClick={() => setIndex((i) => Math.max(0, i - 1))}>
                {chrome.back}
              </button>
            )}
            <button
              type="button"
              className="btn-pi"
              onClick={() => {
                if (isLast) onClose();
                else setIndex((i) => i + 1);
              }}
            >
              {isLast ? chrome.done : chrome.next}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
