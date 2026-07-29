import content from "./content.json";

export type Lang = "en" | "hi" | "mr" | "pa" | "te";
export type ClassName = "clean" | "rice_weevil" | "lesser_grain_borer" | "red_flour_beetle";

export const LANGUAGES = content.LANGUAGES as Record<Lang, string>;
export const CLASS_NAMES: ClassName[] = [
  "clean",
  "rice_weevil",
  "lesser_grain_borer",
  "red_flour_beetle",
];

const PEST_NAMES = content.PEST_NAMES as Record<Lang, Record<string, string>>;
const ADVISORIES = content.ADVISORIES as Record<Lang, Record<string, string>>;
const UI_STRINGS = content.UI_STRINGS as Record<Lang, Record<string, string>>;

function noEmDash(text: string): string {
  return text.replace(/\u2014/g, ", ").replace(/\u2013/g, "-");
}

export function getUi(lang: Lang, key: string): string {
  return noEmDash(UI_STRINGS[lang]?.[key] ?? UI_STRINGS.en[key] ?? key);
}

export function getAdvisory(lang: Lang, className: string): string {
  return noEmDash(ADVISORIES[lang]?.[className] ?? ADVISORIES.en[className] ?? "");
}

export function getPestName(lang: Lang, className: string): string {
  return noEmDash(PEST_NAMES[lang]?.[className] ?? PEST_NAMES.en[className] ?? className);
}
