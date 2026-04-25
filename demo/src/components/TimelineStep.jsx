export default function TimelineStep({ step, label, status }) {
  // status: 'done' | 'active' | 'idle'
  return (
    <div className={`rt-step ${status}`}>
      <div className="rt-dot">{status === 'done' ? '✓' : step}</div>
      <div className="rt-label">{label}</div>
    </div>
  );
}
