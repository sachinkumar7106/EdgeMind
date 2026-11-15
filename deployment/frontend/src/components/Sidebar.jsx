
export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <aside className="sidebar">
      <a className="brand" href="#home">
        <span className="brand-icon">E</span>
        <span>
          <strong>EdgeMind</strong>
          <small>EDGE AI LAB</small>
        </span>
      </a>

      <p className="nav-label">WORKSPACE</p>

      <button
        className={`nav-item ${activeTab === "overview" ? "active" : ""}`}
        onClick={() => setActiveTab("overview")}
      >
        <span>◫</span> Overview
      </button>

      <button
        className={`nav-item ${activeTab === "classify" ? "active" : ""}`}
        onClick={() => setActiveTab("classify")}
      >
        <span>⌁</span> Image classification
      </button>

      <button
        className={`nav-item ${activeTab === "models" ? "active" : ""}`}
        onClick={() => setActiveTab("models")}
      >
        <span>▦</span> Model comparison
      </button>


      <button
        className={`nav-item ${activeTab === "evaluation" ? "active" : ""}`}
        onClick={() => setActiveTab("evaluation")}
      >
        <span>▤</span> Model Evaluation
      </button>

      <div className="sidebar-bottom">
        <div className="project-status">
          <span className="status-dot" />
          <span>
            <strong>Project status</strong>
            <small>Experiments completed</small>
          </span>
        </div>

        <p>Final Year Project · 2026</p>
      </div>
    </aside>
  );
}
