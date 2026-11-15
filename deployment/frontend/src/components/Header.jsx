
export default function Header({ activeTab }) {
  const pageTitles = {
    overview: "Model overview",
    classify: "Image classification",
    models: "Model comparison",
  };

  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">RESEARCH WORKSPACE / EDGEMIND</p>
        <h1>{pageTitles[activeTab] || "Model overview"}</h1>
      </div>

      <div className="topbar-tag">
        <span className="status-dot" />
        CIFAR-10 · 10 classes
      </div>
    </header>
  );
}
