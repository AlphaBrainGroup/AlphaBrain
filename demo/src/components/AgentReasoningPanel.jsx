/**
 * AgentReasoningPanel
 * -------------------
 * Renders the step-by-step reasoning log from a SelfImprovementAgent run.
 *
 * Props:
 *   agentRun   — AgentRunState object (from /api/demo/agent or /api/agent/:id/status)
 *   onAdvance  — callback fired when the user clicks "Next Step" (demo mode)
 */

const STEP_ICONS = {
  eval:       '📊',
  diagnose:   '🔬',
  plan:       '📋',
  dispatch:   '🚀',
  verify:     '✅',
  memory:     '💾',
  stop_check: '🛑',
};

const STATUS_CLASS = {
  pending:   'idle',
  running:   'active',
  completed: 'done',
  failed:    'error',
  skipped:   'idle',
};

function StepRow({ step }) {
  const icon  = STEP_ICONS[step.step_type] || '•';
  const cls   = STATUS_CLASS[step.status] || 'idle';
  return (
    <div className={`agent-step ${cls}`}>
      <span className="agent-step-icon">{icon}</span>
      <div className="agent-step-body">
        <div className="agent-step-label">{step.label}</div>
        {step.message && step.status !== 'pending' && (
          <div className="agent-step-msg">{step.message}</div>
        )}
      </div>
      <div className={`agent-step-badge ${cls}`}>
        {step.status === 'completed' ? '✓' : step.status === 'running' ? '…' : step.status}
      </div>
    </div>
  );
}

export default function AgentReasoningPanel({ agentRun, onAdvance }) {
  if (!agentRun) {
    return (
      <div className="card">
        <div className="section-title">Agent Reasoning</div>
        <p className="card-sub">Start Auto-Improve to see the agent reasoning step by step.</p>
      </div>
    );
  }

  const { iterations = [], status, stop_reason, reasoning_log = [], current_iteration } = agentRun;
  const isDone = status === 'completed' || status === 'stopped';

  return (
    <div className="card agent-panel">
      {/* Header row */}
      <div className="agent-panel-header">
        <div>
          <div className="section-title">Agent Reasoning</div>
          <div className="card-sub">
            {isDone
              ? (stop_reason || 'Run complete.')
              : `Loop ${current_iteration} of ${iterations.length} — ${status}`}
          </div>
        </div>
        {!isDone && onAdvance && (
          <button className="btn-secondary" onClick={onAdvance}>
            Next Step ›
          </button>
        )}
      </div>

      {/* Per-loop step lists */}
      {iterations.map((loop) => (
        <div key={loop.iteration_index} className="agent-loop-block">
          <div className="agent-loop-header">
            <span className={`loop-badge ${loop.status}`}>Loop {loop.iteration_index}</span>
            <span className="agent-loop-meta">
              {loop.patch_cluster} · {loop.patch_strategy}
              {loop.eval_after != null && (
                <span className="agent-loop-uplift">
                  {' '}· {Math.round(loop.eval_before * 100)}%{' '}
                  <span style={{ color: 'var(--green)' }}>→ {Math.round(loop.eval_after * 100)}%</span>
                </span>
              )}
            </span>
          </div>
          <div className="agent-step-list">
            {loop.steps.map((step) => (
              <StepRow key={step.step_id} step={step} />
            ))}
          </div>
        </div>
      ))}

      {/* Raw reasoning log (collapsible) */}
      {reasoning_log.length > 0 && (
        <details className="agent-log-details">
          <summary className="agent-log-summary">Full reasoning log ({reasoning_log.length} entries)</summary>
          <div className="console-list" style={{ marginTop: 8 }}>
            {reasoning_log.map((entry, i) => (
              <div key={i} className="console-item log">{entry}</div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
