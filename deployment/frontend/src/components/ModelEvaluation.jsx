
const CLASS_METRICS = [
  { name: "Airplane", precision: 60.90, recall: 77.10, f1: 68.05 },
  { name: "Automobile", precision: 75.20, recall: 74.90, f1: 75.05 },
  { name: "Bird", precision: 58.40, recall: 45.90, f1: 51.40 },
  { name: "Cat", precision: 48.07, recall: 37.30, f1: 42.00 },
  { name: "Deer", precision: 54.32, recall: 61.60, f1: 57.73 },
  { name: "Dog", precision: 55.20, recall: 44.60, f1: 49.34 },
  { name: "Frog", precision: 59.41, recall: 82.10, f1: 68.93 },
  { name: "Horse", precision: 69.37, recall: 68.40, f1: 68.88 },
  { name: "Ship", precision: 83.41, recall: 68.90, f1: 75.47 },
  { name: "Truck", precision: 69.71, recall: 72.50, f1: 71.08 },
];

export default function ModelEvaluation() {
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">EXPERIMENT ANALYSIS</p>
          <h2>Model Evaluation</h2>
          <p className="section-description">
            Analyze the classification performance of the distilled CNN
            student using its CIFAR-10 confusion matrix and class metrics.
          </p>
        </div>
        <span className="model-pill">CIFAR-10</span>
      </div>

      {/* Confusion Matrix */}
      <div className="evaluation-card">
        <h3>Distilled CNN — Confusion Matrix</h3>
        <p className="section-description">
          Rows represent actual classes and columns represent predicted
          classes. Diagonal cells indicate correct predictions.
        </p>

        <div className="confusion-matrix-container">
          <img
            className="confusion-matrix-image"
            src="/confusion_matrix.png"
            alt="Confusion matrix for the EdgeMind distilled CNN on CIFAR-10"
          />
        </div>

        <p className="muted">
          Off-diagonal cells indicate misclassifications between classes.
        </p>
      </div>

      {/* Per-Class Metrics */}
      <div className="evaluation-card">
        <h3>Per-Class Classification Metrics</h3>
        <p className="section-description">
          Precision measures prediction correctness, recall measures how
          many actual examples were identified, and F1-score balances both.
        </p>

        <div style={{ overflowX: "auto" }}>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-score</th>
              </tr>
            </thead>
            <tbody>
              {CLASS_METRICS.map((item) => (
                <tr key={item.name}>
                  <td>{item.name}</td>
                  <td>{item.precision.toFixed(2)}%</td>
                  <td>{item.recall.toFixed(2)}%</td>
                  <td>{item.f1.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="evaluation-summary">
          <div>
            <span>Test accuracy</span>
            <strong>63.33%</strong>
          </div>
          <div>
            <span>Macro F1-score</span>
            <strong>62.79%</strong>
          </div>
        </div>
      </div>
    </section>
  );
}
