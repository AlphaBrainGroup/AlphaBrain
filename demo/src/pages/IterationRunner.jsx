import { useEffect } from 'react';
import TimelineStep from '../components/TimelineStep';
import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

const STEPS = ['Load checkpoint', 'Dispatch backend job', 'Poll job status', 'Compare results'];

export default function IterationRunner() {
  const { data, demoState, pollIterationStatus } = useNvexRuntime();
  const status = demoState?.iteration_job?.status || 'completed';

  useEffect(() => {
    if (status === 'running' || status === 'queued') {
      const interval = window.setInterval(() => {
        pollIterationStatus();
      }, 2000);
      return () => window.clearInterval(interval);
    }
    return undefined;
  }, [status, pollIterationStatus]);

  return (
    <section className="page-shell page-enter">
      <div>
        <h1 className="page-title">Iteration Runner</h1>
        <p className="page-subtitle">Execution timeline and live training log for the current patch cycle.</p>
      </div>

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