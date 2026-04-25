import TimelineStep from '../components/TimelineStep';
import { D } from '../data/mockData.js';

const STEPS = [
  'Load checkpoint',
  'Train CL patch',
  'Run robustness eval',
  'Promote checkpoint',
];

export default function IterationRunner() {
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
              status={index < 3 ? 'done' : 'active'}
            />
          ))}
        </div>
      </div>

      <div className="card">
        <div className="section-title">Console Output</div>
        <div className="console-list">
          {D.consoleLogs.map((entry) => (
            <div key={`${entry.ts}-${entry.text}`} className={`console-item ${entry.type}`}>
              [{entry.ts}] {entry.text}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}