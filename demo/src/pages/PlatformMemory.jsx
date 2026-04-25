import AssetCard from '../components/AssetCard';
import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

export default function PlatformMemory() {
  const { data } = useNvexRuntime();
  const { stats, recipes, templates, failures } = data.memory;

  return (
    <section className="page-shell page-enter">
      <div>
        <h1 className="page-title">Platform Memory</h1>
        <p className="page-subtitle">Reusable recipes, templates, and failure patterns accumulated from prior loops.</p>
      </div>

      <div className="card-grid-4">
        <AssetCard label="Recipes" value={stats.recipes} tone="cyan" sub="Reusable patch patterns" />
        <AssetCard label="Templates" value={stats.templates} tone="blue" sub="Repeatable pipelines" />
        <AssetCard label="Failure Patterns" value={stats.patterns} tone="red" sub="Known ontology nodes" />
        <AssetCard label="Projects" value={stats.projects} tone="green" sub="Projects contributing memory" />
      </div>

      <div className="memory-grid">
        <div className="card">
          <div className="section-title">Recipes</div>
          <div className="pill-row">
            {recipes.map((item) => <span key={item} className="pill">{item}</span>)}
          </div>
        </div>
        <div className="card">
          <div className="section-title">Templates</div>
          <div className="pill-row">
            {templates.map((item) => <span key={item} className="pill">{item}</span>)}
          </div>
        </div>
        <div className="card" style={{ gridColumn: '1 / -1' }}>
          <div className="section-title">Known Failure Ontology</div>
          <div className="pill-row">
            {failures.map((item) => <span key={item} className="pill">{item}</span>)}
          </div>
        </div>
      </div>
    </section>
  );
}