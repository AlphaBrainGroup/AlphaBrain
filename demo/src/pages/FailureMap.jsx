import FailureCluster from '../components/FailureCluster';
import RadarChart from '../components/RadarChart';
import { D } from '../data/mockData.js';

export default function FailureMap() {
  return (
    <section className="page-shell page-enter">
      <div>
        <h1 className="page-title">Failure Map</h1>
        <p className="page-subtitle">Clustered benchmark failures and task-level risk concentration.</p>
      </div>

      <div className="two-col-grid">
        <div className="card">
          <div className="section-title">Failure Clusters</div>
          <div className="cluster-grid">
            {D.clusters.map((cluster) => (
              <FailureCluster key={cluster.id} cluster={cluster} />
            ))}
          </div>
        </div>

        <div className="card">
          <div className="section-title">Task Breakdown</div>
          <div className="row-gap-4" style={{ alignItems: 'flex-start', justifyContent: 'space-between' }}>
            <RadarChart data={D.taskBreakdown} />
            <div className="flex-1">
              {D.taskBreakdown.map((task) => (
                <div key={task.name} className="bar-item">
                  <div className="bar-label-row">
                    <span className="bar-label-name">{task.name}</span>
                    <span className="bar-label-pct">{task.pct}%</span>
                  </div>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${task.pct}%`, background: task.color }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}