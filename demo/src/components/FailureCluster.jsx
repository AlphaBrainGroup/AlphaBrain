const SEV_BADGE = {
  critical: 'badge-red',
  high:     'badge-orange',
  medium:   'badge-yellow',
  low:      'badge-blue',
};

export default function FailureCluster({ cluster }) {
  const { id, label, pct, count, color, sev } = cluster;
  return (
    <div className="cluster-card">
      <div className="cluster-header">
        <div className="cluster-id" style={{ background: `${color}22`, color }}>
          {id}
        </div>
        <div className="cluster-name">{label}</div>
        <span className={`badge ${SEV_BADGE[sev] || 'badge-blue'}`} style={{ marginLeft: 'auto' }}>
          {sev}
        </span>
      </div>
      <div className="cluster-pct" style={{ color }}>{pct}%</div>
      <div className="cluster-count">{count} episodes</div>
      <div className="cluster-bar">
        <div className="cluster-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}
