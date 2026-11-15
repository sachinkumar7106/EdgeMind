
import { useRef, useState } from "react";

const CLASSES = [
  "airplane",
  "automobile",
  "bird",
  "cat",
  "deer",
  "dog",
  "frog",
  "horse",
  "ship",
  "truck",
];

export default function ImageClassifier() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);

  function selectImage(selected) {
    if (!selected) return;

    if (!selected.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }

    if (selected.size > 5 * 1024 * 1024) {
      setError("Please choose an image smaller than 5 MB.");
      return;
    }

    setPreview((oldPreview) => {
      if (oldPreview) URL.revokeObjectURL(oldPreview);
      return URL.createObjectURL(selected);
    });

    setFile(selected);
    setPrediction(null);
    setError("");
  }

  function handleFileChange(event) {
    selectImage(event.target.files?.[0]);
    event.target.value = "";
  }

  function clearImage() {
    if (preview) URL.revokeObjectURL(preview);

    setFile(null);
    setPreview("");
    setPrediction(null);
    setError("");
    setLoading(false);

    if (fileInput.current) {
      fileInput.current.value = "";
    }
  }

  async function handlePredict() {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch(
        "http://127.0.0.1:5000/api/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction request failed.");
      }

      setPrediction(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to Flask. Check that the backend is running on port 5000."
          : err.message || "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  }

  const sortedProbabilities = prediction?.probabilities
    ? Object.entries(prediction.probabilities)
        .sort((a, b) => b[1] - a[1])
    : [];

  return (
    <section className="panel classifier-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">INFERENCE WORKSPACE</p>
          <h2>Classify an image</h2>
          <p className="section-description">
            Upload an image and let the EdgeMind distilled CNN predict its class.
          </p>
        </div>
        <span className="model-pill">Distilled CNN</span>
      </div>

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        hidden
      />

      <div
        className={`upload-zone ${preview ? "has-image" : ""}`}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          selectImage(event.dataTransfer.files?.[0]);
        }}
      >
        {preview ? (
          <>
            <img
              className="image-preview"
              src={preview}
              alt="Selected upload"
            />
            <strong>{file?.name}</strong>
            <span className="muted">
              {file ? (file.size / (1024 * 1024)).toFixed(2) : "0"} MB
            </span>
            <button
              className="secondary-button"
              onClick={clearImage}
              disabled={loading}
            >
              Remove image
            </button>
          </>
        ) : (
          <>
            <div className="upload-icon">↑</div>
            <h3>Drop an image here</h3>
            <p>or choose a file from your device</p>
            <button
              className="primary-button"
              onClick={() => fileInput.current?.click()}
            >
              Choose image
            </button>
            <span className="muted">Image files · Maximum 5 MB</span>
          </>
        )}
      </div>

      <div className="prediction-actions">
        <button
          className="primary-button"
          onClick={handlePredict}
          disabled={!file || loading}
        >
          {loading ? "Classifying image..." : "Run classification"}
          {!loading && <span> →</span>}
        </button>
      </div>

      {loading && (
        <div className="notice" role="status">
          <strong>Analyzing image...</strong>
          <p>The distilled CNN is processing your image.</p>
        </div>
      )}

      {error && (
        <div className="notice" role="alert">
          <strong>Classification failed</strong>
          <p>{error}</p>
        </div>
      )}

      {prediction && (
        <div className="prediction-result">
          <div className="result-heading">
            <div>
              <p className="eyebrow">MODEL OUTPUT</p>
              <h2>
                {prediction.prediction.charAt(0).toUpperCase() +
                  prediction.prediction.slice(1)}
              </h2>
              <p className="result-confidence">
                {prediction.confidence}% confidence
              </p>
            </div>
            <div className="result-badge">Top prediction</div>
          </div>

          <p className="result-explanation">
            Confidence is the model's estimated probability for its predicted
            class, not a guarantee of correctness.
          </p>

          <div className="probability-section">
            <div className="probability-heading">
              <h3>Class probabilities</h3>
              <span>All 10 classes</span>
            </div>

            {sortedProbabilities.length > 0 ? (
              <div className="probability-list">
                {sortedProbabilities.map(([className, probability]) => {
                  const isPredicted =
                    className === prediction.prediction;

                  return (
                    <div className="probability-row" key={className}>
                      <div className="probability-label">
                        <span className={isPredicted ? "predicted-label" : ""}>
                          {className.charAt(0).toUpperCase() +
                            className.slice(1)}
                        </span>
                        <strong>{Number(probability).toFixed(2)}%</strong>
                      </div>

                      <div
                        className="probability-track"
                        role="progressbar"
                        aria-label={`${className} probability`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={Number(probability)}
                      >
                        <div
                          className={`probability-fill ${
                            isPredicted ? "predicted-fill" : ""
                          }`}
                          style={{
                            width: `${Math.min(
                              100,
                              Math.max(0, Number(probability) || 0)
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="muted">
                Probability details are not available. Make sure the Flask API
                returns the probabilities field.
              </p>
            )}
          </div>
        </div>
      )}

      <div className="supported-classes">
        <h3>Supported CIFAR-10 classes</h3>
        <div className="class-chips">
          {CLASSES.map((name) => (
            <span key={name}>
              {name.charAt(0).toUpperCase() + name.slice(1)}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
