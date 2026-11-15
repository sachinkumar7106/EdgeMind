
export default function MetricCard({ label, value, note, accent }) {
  return (
    <article className="metric-card">
      <div className={`metric-mark ${accent}`} />
      <p>{label}</p>
      <h3>{value}</h3>
      <span>{note}</span>
    </article>
  );
}
