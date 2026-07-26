const BLOCKS: { label: string; body: string }[] = [
  {
    label: "Problem",
    body: "India stores over 80 million tonnes of food grain. Insects in storage cause about 1,300 crore rupees in annual loss (IGMRI). Smallholders often find damage only after 10 to 20 percent of the grain is already affected, because hand and smell checks miss early activity.",
  },
  {
    label: "Solution",
    body: "Farmers record grain sounds by holding their phone against the bag or bin. Kaan converts the audio to a mel spectrogram and runs a compact CNN that classifies clean grain, rice weevil, lesser grain borer, and red flour beetle. It returns the pest class, confidence, a simple advisory, and an accessibility-safe result using both symbol and text.",
  },
  {
    label: "Why AI",
    body: "These pests produce overlapping sounds in the 300 to 4000 Hz range. Rule-based thresholds cannot separate them reliably. A trained CNN learns the subtle spectral patterns needed for species-level screening.",
  },
  {
    label: "Impact and inclusion",
    body: "Available in English, Hindi, Marathi, Punjabi, and Telugu. Designed for low-resource settings using phone mic, offline classification, and a guided tour. Aimed at farmers, FPOs, and Krishi Vigyan Kendras as a screening aid. Open pipeline so local agencies can retrain on regional audio.",
  },
  {
    label: "Ethics and limits",
    body: "Reports confidence and can stay uncertain instead of guessing. Phone quality, noise, and early low-density infestations still need field validation. Pulse beetle and legume detection are future work.",
  },
];

const TECH = [
  "Validation accuracy 97.76 percent, macro F1 0.98, after leakage-aware retraining",
  "INT8 model runs in the browser with no cloud needed for inference",
  "Built on open IRRI acoustic research data, released under MIT",
  "Live at kaan-web.vercel.app",
];

export default function SummitSummary() {
  return (
    <section className="mt-16 pt-10 border-t border-black/20 text-left" aria-labelledby="summit-heading">
      <p className="panel-label mb-2">Recognition</p>
      <h2 id="summit-heading" className="text-2xl font-bold mb-3">
        Kaan at the AI Impact Summit
      </h2>
      <p className="text-base leading-relaxed mb-8">
        Kaan is an AI-powered acoustic detector that helps Indian farmers catch stored-grain insect
        infestation early using a normal phone, offline, in their language.
      </p>

      <div>
        {BLOCKS.map((block, i) => (
          <div className="pi-timeline-item" key={block.label}>
            <div className="pi-timeline-row">
              <span className="pi-timeline-title">{block.label}</span>
              <span className="pi-timeline-meta">{String(i + 1).padStart(2, "0")}</span>
            </div>
            <p className="pi-timeline-body">{block.body}</p>
          </div>
        ))}

        <div className="pi-timeline-item">
          <div className="pi-timeline-row">
            <span className="pi-timeline-title">Technical highlights</span>
            <span className="pi-timeline-meta">{String(BLOCKS.length + 1).padStart(2, "0")}</span>
          </div>
          <ul className="pi-timeline-body list-disc pl-5 space-y-1">
            {TECH.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="panel">
        <p className="panel-label mb-2">Summit pitch</p>
        <p className="text-sm leading-relaxed">
          Kaan listens to grain so farmers do not have to wait until they can see the damage. Open,
          multilingual, offline AI for food security and farmer livelihoods.
        </p>
      </div>
    </section>
  );
}
