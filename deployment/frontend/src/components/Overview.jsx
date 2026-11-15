
import MetricCard from "./MetricCard";
import PipelineStep from "./PipelineStep";

const models = [
  {
    type: "Teacher model",
    accuracy: 65.57,
  },
  {
    type: "Baseline student",
    accuracy: 55.06,
  },
  {
    type: "Distilled student",
    accuracy: 63.33,
  },
];

export default function Overview({ onClassify, onCompare }) {
  return (
    <>
      <section className="hero">
        <div className="hero-content">
          <span className="hero-label">KNOWLEDGE DISTILLATION</span>

          <h2>
            Transformer intelligence.
            <br />
            <span>Lightweight execution.</span>
          </h2>

          <p>
            Evaluating whether a compact CNN can learn from a
            Vision Transformer and bring its classification
            performance to resource-constrained environments.
          </p>

          <button className="primary-button" onClick={onClassify}>
            Explore image classifier <span>→</span>
          </button>
        </div>

        <div className="hero-art" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="core">
            <span>EM</span>
          </div>
          <div className="art-chip chip-one">ViT</div>
          <div className="art-chip chip-two">CNN</div>
          <div className="art-chip chip-three">KD</div>
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard
          label="Distilled test accuracy"
          value="63.33%"
          note="10,000 test images"
          accent="purple"
        />

        <MetricCard
          label="Gain over baseline"
          value="+8.27 pp"
          note="Percentage-point improvement"
          accent="green"
        />

        <MetricCard
          label="Student parameters"
          value="1.53M"
          note="MobileNetV3-Small"
          accent="blue"
        />

        <MetricCard
          label="Model weight size"
          value="5.88 MB"
          note="FP32 state tensors"
          accent="orange"
        />
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">EXPERIMENTAL RESULTS</p>
            <h2>Accuracy comparison</h2>
          </div>

          <button className="text-button" onClick={onCompare}>
            View details →
          </button>
        </div>

        <div className="accuracy-chart">
          {models.map((model) => (
            <div className="bar-row" key={model.type}>
              <div className="bar-label">
                <span>{model.type}</span>
                <strong>{model.accuracy.toFixed(2)}%</strong>
              </div>

              <div className="bar-track">
                <div
                  className={`bar-fill ${
                    model.type.startsWith("Distilled")
                      ? "highlight"
                      : ""
                  }`}
                  style={{ width: `${model.accuracy}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        <p className="chart-footnote">
          Measured test accuracy on CIFAR-10. Results are from your
          saved experiment outputs.
        </p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">PIPELINE</p>
            <h2>From teacher to student</h2>
          </div>
        </div>

        <div className="pipeline">
          <PipelineStep
            number="01"
            title="Vision Transformer"
            description="Trained teacher"
          />

          <span className="pipeline-arrow">→</span>

          <PipelineStep
            number="02"
            title="Knowledge distillation"
            description="Soft targets + labels"
          />

          <span className="pipeline-arrow">→</span>

          <PipelineStep
            number="03"
            title="MobileNetV3-Small"
            description="Distilled student"
          />
        </div>
      </section>
    </>
  );
}
