import MultiIterationChart from '../components/MultiIterationChart';
import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

export default function ImprovementReport() {
  const { data, agentRun } = useNvexRuntime();
  const { before, after, changes, assets, nextIter } = data.report;
  const uplift = after.success - before.success;

  // Multi-loop data from agent run (if available)
  const loopIterations = agentRun?.iterations?.filter((l) => l.eval_after != null) ?? [];
  const hasMultiLoop   = loopIterations.length > 1;
  const targetKpi      = agentRun?.target_kpi ?? 0.75;
  const stopReason     = agentRun?.stop_reason;

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

      {/* Multi-loop chart — shown only when agent has run ≥2 loops */}
      {hasMultiLoop && (
        <MultiIterationChart iterations={loopIterations} targetKpi={targetKpi} />
      )}

      {/* Agent stop reason callout */}
      {stopReason && (
        <div className="card agent-stop-card">
          <div className="section-title" style={{ color: 'var(--green)' }}>Why the agent stopped</div>
          <p className="card-sub" style={{ marginTop: 6 }}>{stopReason}</p>
        </div>
      )}

      <div className="two-col-grid">
        <div className="card">
          <div className="section-title">What Changed</div>
          <div className="report-list">
            {changes.map((item) => (
              <div key={item} className="report-item">{item}</div>
            ))}
            {/* Inject per-loop summaries when multi-loop is active */}
            {hasMultiLoop && loopIterations.map((loop) => (
              <div key={loop.iteration_index} className="report-item">
                Loop {loop.iteration_index}: {loop.patch_cluster} patch →{' '}
                {Math.round(loop.eval_before * 100)}%{' '}
                <span style={{ color: 'var(--green)' }}>→ {Math.round(loop.eval_after * 100)}%</span>
              </div>
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
            {/* Recipe chips from agent loops */}
            {hasMultiLoop && loopIterations.map((loop) => (
              <span key={`recipe-${loop.iteration_index}`} className="asset-chip recipe">
                recipe: {loop.patch_strategy}_loop{loop.iteration_index}
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
