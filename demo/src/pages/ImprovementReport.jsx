import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

export default function ImprovementReport() {
  const { data } = useNvexRuntime();
  const { before, after, changes, assets, nextIter } = data.report;
  const uplift = after.success - before.success;

  return (
    <section className="page-shell page-enter">
      <div>
        <h1 className="page-title">Improvement Report</h1>
        <p className="page-subtitle">Before-and-after results from the completed patch loop.</p>
      </div>

      <div className="card-grid-3">
        <div className="card"><div className="card-title">Before</div><div className="card-value">{before.success}%</div><div className="card-sub">{before.clusters} failure clusters</div></div>
        <div className="card"><div className="card-title">After</div><div className="card-value" style={{ color: 'var(--green)' }}>{after.success}%</div><div className="card-sub">{after.clusters} failure clusters</div></div>
        <div className="card"><div className="card-title">Uplift</div><div className="card-value" style={{ color: 'var(--green)' }}>+{uplift}pp</div><div className="card-sub">Recovery score {before.recovery}% → {after.recovery}%</div></div>
      </div>

      <div className="two-col-grid">
        <div className="card">
          <div className="section-title">What Changed</div>
          <div className="report-list">
            {changes.map((item) => (
              <div key={item} className="report-item">{item}</div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="section-title">Assets Created</div>
          <div className="assets-created">
            {assets.map((asset) => (
              <span key={asset.name} className={`asset-chip ${asset.type}`}>
                {asset.type}: {asset.name}
              </span>
            ))}
          </div>
          <div className="divider" />
          <div className="card-sub">Next iteration target: {nextIter}</div>
        </div>
      </div>
    </section>
  );
}