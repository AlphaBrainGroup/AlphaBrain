import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

export default function PatchPlan() {
  const { data } = useNvexRuntime();
  const { patchPlan } = data;

  return (
    <section className="page-shell page-enter">
      <div>
        <h1 className="page-title">Patch Plan</h1>
        <p className="page-subtitle">Targeted intervention proposed from the current failure diagnosis.</p>
      </div>

      <div className="two-col-grid">
        <div className="card">
          <div className="section-title">Capability Gaps</div>
          <div className="plan-list">
            {patchPlan.gaps.map((gap) => (
              <div key={gap} className="plan-item">{gap}</div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="section-title">Execution Outline</div>
          <div className="plan-list">
            <div className="plan-item">Collect {patchPlan.data.episodes} patch episodes and {patchPlan.data.corrections} correction trajectories.</div>
            <div className="plan-item">Run {patchPlan.training.strategy} with {patchPlan.data.modality} for roughly {patchPlan.training.duration}.</div>
            <div className="plan-item">Verify on {patchPlan.verify.suite} using {patchPlan.verify.env} before checkpoint promotion.</div>
          </div>
        </div>
      </div>

      <div className="card-grid-4">
        <div className="card"><div className="card-title">Data Mix</div><div className="card-sub">{patchPlan.data.ratio}</div></div>
        <div className="card"><div className="card-title">Strategy</div><div className="card-sub">{patchPlan.training.note}</div></div>
        <div className="card"><div className="card-title">Expected Uplift</div><div className="card-sub">+{patchPlan.uplift.lo} to +{patchPlan.uplift.hi} points</div></div>
        <div className="card"><div className="card-title">Confidence</div><div className="card-sub">{Math.round(patchPlan.uplift.confidence * 100)}%</div></div>
      </div>
    </section>
  );
}