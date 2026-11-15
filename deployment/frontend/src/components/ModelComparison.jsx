
const models = [
  {
    name: "Vision Transformer",
    type: "Teacher model",
    accuracy: 65.57,
    parameters: "1,806,538",
    weights: "6.89 MB",
    checkpoint: "20.77 MB",
    cpu: "6.716 ms",
    gpu: "4.186 ms",
  },
  {
    name: "MobileNetV3-Small",
    type: "Baseline student",
    accuracy: 55.06,
    parameters: "1,528,106",
    weights: "5.88 MB",
    checkpoint: "17.73 MB",
    cpu: "5.862 ms",
    gpu: "7.641 ms",
  },
  {
    name: "MobileNetV3-Small",
    type: "Distilled student",
    accuracy: 63.33,
    parameters: "1,528,106",
    weights: "5.88 MB",
    checkpoint: "17.74 MB",
    cpu: "6.268 ms",
    gpu: "8.008 ms",
  },
];

export default function ModelComparison() {
  const teacher = models[0];
  const baseline = models[1];
  const distilled = models[2];

  const accuracyGain = (
    distilled.accuracy - baseline.accuracy
  ).toFixed(2);

  return (
    <section className="panel model-comparison">
      <div className="section-heading">
        <div>
          <p className="eyebrow">MEASURED BENCHMARKS</p>
          <h2>Teacher vs Student Models</h2>
          <p className="section-description">
            Compare accuracy, model complexity, storage requirements,
            and batch-one inference latency.
          </p>
        </div>
      </div>

      {/* Summary cards */}
      <div className="comparison-cards">
        {models.map((model) => (
          <div className="comparison-card" key={model.type}>
            <p>{model.type}</p>
            <h3>{model.accuracy.toFixed(2)}%</h3>
            <span>Test accuracy</span>
            <div className="accuracy-track">
              <div
                className="accuracy-fill"
                style={{ width: `${model.accuracy}%` }}
              />
            </div>
            <small>{model.name}</small>
          </div>
        ))}
      </div>

      {/* Accuracy improvement */}
      <div className="improvement-card">
        <div>
          <p className="eyebrow">KNOWLEDGE DISTILLATION RESULT</p>
          <h3>+{accuracyGain} percentage points</h3>
          <p>
            Improvement in test accuracy of the distilled CNN
            compared with the baseline CNN.
          </p>
        </div>
      </div>

      {/* Detailed comparison */}
      <div className="comparison-table-section">
        <h3>Detailed Model Comparison</h3>

        <div className="table-wrap">
          <table className="model-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Teacher (ViT)</th>
                <th>Baseline CNN</th>
                <th>Distilled CNN</th>
              </tr>
            </thead>

            <tbody>
              <tr>
                <td>Test accuracy</td>
                <td>{teacher.accuracy.toFixed(2)}%</td>
                <td>{baseline.accuracy.toFixed(2)}%</td>
                <td><strong>{distilled.accuracy.toFixed(2)}%</strong></td>
              </tr>
              <tr>
                <td>Parameters</td>
                <td>{teacher.parameters}</td>
                <td>{baseline.parameters}</td>
                <td>{distilled.parameters}</td>
              </tr>
              <tr>
                <td>Model weights</td>
                <td>{teacher.weights}</td>
                <td>{baseline.weights}</td>
                <td>{distilled.weights}</td>
              </tr>
              <tr>
                <td>Checkpoint size</td>
                <td>{teacher.checkpoint}</td>
                <td>{baseline.checkpoint}</td>
                <td>{distilled.checkpoint}</td>
              </tr>
              <tr>
                <td>CPU latency</td>
                <td>{teacher.cpu}</td>
                <td>{baseline.cpu}</td>
                <td>{distilled.cpu}</td>
              </tr>
              <tr>
                <td>GPU latency</td>
                <td>{teacher.gpu}</td>
                <td>{baseline.gpu}</td>
                <td>{distilled.gpu}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Interpretation */}
      <div className="notice">
        <strong>Key observation</strong>
        <p>
          The baseline and distilled CNN use the same architecture
          and have the same number of parameters. Knowledge
          distillation improves test accuracy from 55.06% to
          63.33%, demonstrating the benefit of teacher guidance.
        </p>
        <p>
          The distilled CNN has a slightly larger checkpoint than
          the baseline. Its measured GPU latency is also higher
          in this benchmark, so we do not claim a universal
          inference-speed improvement.
        </p>
      </div>
    </section>
  );
}
