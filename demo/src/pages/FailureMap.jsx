import FailureCluster from '../components/FailureCluster';
import RadarChart from '../components/RadarChart';
import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

export default function FailureMap() {
  const { data } = useNvexRuntime();

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
            {data.clusters.map((cluster) => (
              <FailureCluster key={cluster.id} cluster={cluster} />
            ))}
          </div>
        </div>

        <div className="card">
          <div className="section-title">Task Breakdown</div>
          <div className="row-gap-4" style={{ alignItems: 'flex-start', justifyContent: 'space-between' }}>
            <RadarChart data={data.taskBreakdown} />
            <div className="flex-1">
              {data.taskBreakdown.map((task) => (
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

      <div className="two-col-grid">
        <div className="card">
          <div className="section-title">Root-Cause Hypotheses</div>
          <div className="root-cause-list">
            {data.rootCauses.map((cause, index) => (
              <div key={cause} className="root-cause-item">
                <div className="rci-num">{index + 1}</div>
                <div className="rci-text">{cause}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="section-title">Representative Episodes</div>
          <div className="episode-grid">
            {data.representativeEpisodes.map((episode) => (
              <div key={episode.id} className="episode-card">
                <div className="episode-thumb">{episode.cluster}</div>
                <div className="episode-label">{episode.label} · {episode.id}</div>
                <div className="episode-meta">Cluster {episode.cluster} · {episode.detail}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card diagnosis-card">
        <div className="diag-label">Nvex Diagnosis</div>
        <div className="diag-text">{data.diagnosis}</div>
      </div>
    </section>
  );
}