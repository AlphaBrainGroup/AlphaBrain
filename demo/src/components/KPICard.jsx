export default function KPICard({ title, value, sub, accentColor, children }) {
  return (
    <div className="card" style={{ borderColor: accentColor ? `${accentColor}40` : undefined }}>
      {accentColor && (
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: accentColor }} />
      )}
      <div className="card-title">{title}</div>
      <div className="card-value" style={accentColor ? { color: accentColor } : undefined}>{value}</div>
      {sub && <div className="card-sub">{sub}</div>}
      {children}
    </div>
  );
}
