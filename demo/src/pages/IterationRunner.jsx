import { useEffect, useState } from 'react';
import AgentReasoningPanel from '../components/AgentReasoningPanel';
import TimelineStep from '../components/TimelineStep';
import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

const STEPS = ['Load checkpoint', 'Dispatch backend job', 'Poll job status', 'Compare results'];

export default function IterationRunner() {
  const { data, demoState, pollIterationStatus, agentRun, startAutoImprove, advanceAgentStep } = useNvexRuntime();
  const status = demoState?.iteration_job?.status || 'completed';
  const [autoMode, setAutoMode] = useState(false);

  // Poll M2 job status
  useEffect(() => {
    if (status === 'running' || status === 'queued') {
      const interval = window.setInterval(() => {
        pollIterationStatus();
      }, 2000);
      return () => window.clearInterval(interval);
    }
    return undefined;
  }, [status, pollIterationStatus]);

  const handleAutoImprove = () => {
    setAutoMode(true);
    startAutoImprove();
  };

  const agentIsDone = agentRun?.status === 'completed' || agentRun?.status === 'stopped';

  // Loop progress derived from agent run
  const totalLoops = agentRun?.iterations?.length ?? 3;
  const doneLoops  = agentRun?.iterations?.filter((l) => l.status === 'completed').length ?? 0;
  const currentKpi = agentRun?.iterations
    ?.filter((l) => l.eval_after != null)
    .slice(-1)[0]?.eval_after ?? null;
  const targetKpi  = agentRun?.target_kpi ?? 0.75;

  return (
    <section className="page-shell page-enter">
      <div className="page-header-row">
        <div>
          <h1 className="page-title">Iteration Runner</h1>
          <p className="page-subtitle">Execution timeline and live training log for the current patch cycle.</p>
        </div>
        {!autoMode && (
          <button className="btn-primary" onClick={handleAutoImprove}>
            ⚡ Auto-Improve
          </button>
        )}
      </div>

      {/* Loop progress bar — only shown in auto mode */}
      {autoMode && (
        <div className="card agent-progress-card">
          <div className="agent-progress-header">
            <div>
              <div className="section-title">Autonomous Loop Progress</div>
              <div className="card-sub">
                {agentIsDone
                  ? (agentRun?.stop_reason || 'Run complete.')
                  : `Loop ${agentRun?.current_iteration ?? 1} / ${totalLoops} running…`}
              </div>
            </div>
            <div className="agent-kpi-display">
              {currentKpi !== null && (
                <>
                  <span className="agent-kpi-val" style={{ color: currentKpi >= targetKpi ? 'var(--green)' : 'var(--t1)' }}>
                    {Math.round(currentKpi * 100)}%
                  </span>
                  <span className="card-sub"> / {Math.round(targetKpi * 100)}% target</span>
                </>
              )}
            </div>
          </div>
          <div className="agent-progress-bar-track">
            <div
              className="agent-progress-bar-fill"
              style={{ width: `${totalLoops > 0 ? (doneLoops / totalLoops) * 100 : 0}%` }}
            />
          </div>
          <div className="agent-progress-loops">
            {Array.from({ length: totalLoops }, (_, i) => {
              const loop = agentRun?.iterations?.[i];
              const cls  = !loop ? 'idle'
                : loop.status === 'completed' ? 'done'
                : loop.status === 'running'   ? 'active'
                : 'idle';
              return (
                <div key={i} className={`agent-loop-dot ${cls}`}>
                  {loop?.status === 'completed' ? '✓' : i + 1}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Manual timeline — shown when not in auto mode */}
      {!autoMode && (
        <div className="card">
          <div className="section-title">Timeline</div>
          <div className="timeline-grid">
            {STEPS.map((label, index) => (
              <TimelineStep
                key={label}
                step={index + 1}
                label={label}
                status={
                  status === 'completed'
                    ? 'done'
                    : status === 'running' && index < 2
                      ? 'done'
                      : status === 'running' && index === 2
                        ? 'active'
                        : index === 0
                          ? 'done'
                          : 'idle'
                }
              />
            ))}
          </div>
        </div>
      )}

      {/* Agent reasoning panel — shown in auto mode */}
      {autoMode && (
        <AgentReasoningPanel
          agentRun={agentRun}
          onAdvance={!agentIsDone ? advanceAgentStep : undefined}
        />
      )}

      {/* Console output */}
      <div className="card">
        <div className="section-title">Console Output</div>
        <div className="console-list">
          {data.consoleLogs.map((entry) => (
            <div key={`${entry.ts}-${entry.text}`} className={`console-item ${entry.type}`}>
              [{entry.ts}] {entry.text}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
