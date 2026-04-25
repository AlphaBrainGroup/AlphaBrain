import KPICard from '../components/KPICard';
import AssetCard from '../components/AssetCard';
import { useNvexRuntime } from '../data/NvexRuntimeContext.jsx';

export default function ProjectOverview() {
  const { data } = useNvexRuntime();
  const { project, rootCauses, availableAssets } = data;

  return (
    <section className="page-shell page-enter">
      <div>
        <h1 className="page-title">Project Overview</h1>
        <p className="page-subtitle">Benchmark context, operating status, and the current intervention target.</p>
      </div>

      <div className="card-grid-3">
        <KPICard title="Project" value={project.name} sub={project.suite} />
        <KPICard title="Status" value={project.status} sub={project.statusNote} accentColor="#f97316" />
        <KPICard title="Next Action" value="CL Patch" sub={project.nextAction} accentColor="#10b981" />
      </div>

      <div className="card">
        <div className="section-title">Root Cause Summary</div>
        <div className="insight-list">
          {rootCauses.map((cause) => (
            <div key={cause} className="insight-item">{cause}</div>
          ))}
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