
import { useState } from "react";
import "./App.css";

import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import Overview from "./components/Overview";
import ImageClassifier from "./components/ImageClassifier";
import ModelComparison from "./components/ModelComparison";
import ModelEvaluation from "./components/ModelEvaluation";

function App() {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="app-shell">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      <main className="main-content" id="home">
        <Header activeTab={activeTab} />

        {activeTab === "overview" && (
          <Overview
            onClassify={() => setActiveTab("classify")}
            onCompare={() => setActiveTab("models")}
          />
        )}

        {activeTab === "classify" && <ImageClassifier />}

        {activeTab === "models" && <ModelComparison />}

        {activeTab === "evaluation" && <ModelEvaluation />}

        <footer className="footer">
          <span>EdgeMind · Final Year Research Project</span>
          <span>Vision Transformer → CNN Knowledge Distillation</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
