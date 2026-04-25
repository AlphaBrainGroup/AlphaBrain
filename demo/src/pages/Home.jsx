import KPICard from '../components/KPICard';
import AssetCard from '../components/AssetCard';
import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

export default function Home({ onNav }) {
  const { data } = useNvexRuntime();
  const { project, featuredValue, recentProjects, availableAssets } = data;

  return (
    <section className="page-shell page-enter">
      <div>
        <h1 className="page-title">Nvex Project Hub</h1>
        <p className="page-subtitle">Failure-to-fix orchestration for physical AI post-training.</p>
      </div>

      <div className="hero-panel">
        <div className="card">
          <div className="section-title">Active Project</div>
          <h2 style={{ fontSize: 28, marginBottom: 8 }}>{project.name}</h2>
          <p style={{ color: 'var(--t2)', maxWidth: 680 }}>
            Nvex has diagnosed a perception-heavy failure pattern and queued the next highest-leverage patch.
          </p>
          <div className="kpi-callout">
            <div className="kpi-callout-value">{project.kpi.successRate}%</div>
            <div className="kpi-callout-copy">Current success rate before patching. The next loop targets occlusion and recovery gaps.</div>
          </div>
          <div className="row-gap-4" style={{ marginTop: 18 }}>
            <button className="btn btn-primary" onClick={() => onNav('failure')}>Inspect Failure Map</button>
            <button className="btn btn-ghost" onClick={() => onNav('runner')}>Run Iteration</button>
          </div>
        </div>

        <div className="panel-stack">
          <KPICard title="Checkpoint" value={project.checkpoint} sub={project.domain} accentColor="#6366f1" />
          <KPICard title="Top Risk" value={project.kpi.topRisk} sub={project.statusNote} accentColor="#f43f5e" />
          <KPICard title="Next Action" value="Patch" sub={project.nextAction} accentColor="#10b981" />
        </div>
      </div>

      <div className="card-grid-4">
        <KPICard title="Success Rate" value={`${project.kpi.successRate}%`} sub="Current benchmark aggregate" />
        <KPICard title="Failure Clusters" value={project.kpi.failClusters} sub="Distinct issue groups" />
        <KPICard title="Confidence" value={`${Math.round(project.kpi.confidence * 100)}%`} sub="Diagnosis confidence" />
        <KPICard title="Platform Assets" value={project.assets.runs + project.assets.recipes + project.assets.templates} sub="Reusable artifacts generated" />
      </div>

      <div className="two-col-grid">
        <div className="card">
          <div className="section-title">Featured Value</div>
          <div className="feature-list">
            {featuredValue.map((item) => (
              <div key={item.title} className="feature-item">
                <div className="feature-name">{item.title}</div>
                <div className="feature-copy">{item.description}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="section-title">Recent Projects</div>
          <div className="project-list">
            {recentProjects.map((entry) => (
              <button key={entry.name} className="project-list-item" onClick={() => onNav(entry.route)}>
                <div>
                  <div className="project-list-name">{entry.name}</div>
                  <div className="project-list-meta">{entry.checkpoint} · {entry.suite}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div className="project-list-score">{entry.successRate}%</div>
                  <div className="project-list-meta">{entry.status}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div>
        <div className="section-title">Available Assets</div>
        <div className="card-grid-4">
          {availableAssets.map((asset) => (
            <AssetCard key={asset.label} {...asset} />
          ))}
        </div>
      </div>
    </section>
  );
}