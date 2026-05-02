const TONE_CLASS = {
  blue: 'badge-blue',
  cyan: 'badge-cyan',
  green: 'badge-green',
  red: 'badge-red',
  orange: 'badge-orange',
  yellow: 'badge-yellow',
};

export default function AssetCard({ label, value, tone = 'blue', sub }) {
  return (
    <div className="asset-card">
      <div className="row-gap-2" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div className="asset-card-label">{label}</div>
        <span className={`badge ${TONE_CLASS[tone] || 'badge-blue'}`}>{tone}</span>
      </div>
      <div className="asset-card-value">{value}</div>
      {sub ? <div className="asset-card-sub">{sub}</div> : null}
    </div>
  );
}