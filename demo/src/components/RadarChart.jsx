export default function RadarChart({ data }) {
  const cx = 120, cy = 110, r = 80;
  const n = data.length;
  const pts = data.map((d, i) => {
    const angle = (i * 2 * Math.PI / n) - Math.PI / 2;
    const fr = (d.pct / 100) * r;
    return {
      x: cx + fr * Math.cos(angle),
      y: cy + fr * Math.sin(angle),
      lx: cx + (r + 18) * Math.cos(angle),
      ly: cy + (r + 18) * Math.sin(angle),
      ...d,
    };
  });

  const polyPoints = pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');

  // Grid circles
  const grids = [0.25, 0.5, 0.75, 1].map(f => ({
    r: r * f,
    cx, cy,
  }));

  return (
    <svg viewBox="0 0 240 220" width="240" height="220">
      {grids.map((g, i) => (
        <circle key={i} cx={g.cx} cy={g.cy} r={g.r} fill="none" stroke="rgba(99,102,241,0.1)" strokeWidth="1" />
      ))}
      {pts.map((p, i) => {
        const angle = (i * 2 * Math.PI / n) - Math.PI / 2;
        return (
          <line key={i}
            x1={cx} y1={cy}
            x2={(cx + r * Math.cos(angle)).toFixed(1)}
            y2={(cy + r * Math.sin(angle)).toFixed(1)}
            stroke="rgba(99,102,241,0.12)" strokeWidth="1"
          />
        );
      })}
      <polygon
        points={polyPoints}
        fill="rgba(244,63,94,0.15)"
        stroke="#f43f5e"
        strokeWidth="1.5"
      />
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={p.x.toFixed(1)} cy={p.y.toFixed(1)} r="3" fill="#f43f5e" />
          <text
            x={p.lx.toFixed(1)} y={p.ly.toFixed(1)}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="var(--t2)"
            fontSize="9"
            fontFamily="Inter"
          >
            {p.name.replace('libero_', '')}
          </text>
        </g>
      ))}
    </svg>
  );
}
