
export default function PipelineStep({ number, title, description }) {
  return (
    <div className="pipeline-step">
      <span className="step-number">{number}</span>
      <strong>{title}</strong>
      <small>{description}</small>
    </div>
  );
}
