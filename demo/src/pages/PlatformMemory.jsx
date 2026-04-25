import { D } from '../data/mockData.js';

export default function PlatformMemory() {
  const { stats, recipes, templates, failures } = D.memory;

  return (
    <section className="page-shell page-enter">
      <div>
        <h1 className="page-title">Platform Memory</h1>
        <p className="page-subtitle">Reusable recipes, templates, and failure patterns accumulated from prior loops.</p>
      </div>

      <div className="card-grid-4">
        <div className="card"><div className="card-title">Recipes</div><div className="card-value">{stats.recipes}</div></div>
        <div className="card"><div className="card-title">Templates</div><div className="card-value">{stats.templates}</div></div>
        <div className="card"><div className="card-title">Failure Patterns</div><div className="card-value">{stats.patterns}</div></div>
        <div className="card"><div className="card-title">Projects</div><div className="card-value">{stats.projects}</div></div>
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