/**
 * MultiIterationChart
 * --------------------
 * A pure-CSS/SVG bar chart showing success rate across agent loop iterations.
 * No external charting library required.
 *
 * Props:
 *   iterations  — array of { iteration_index, eval_before, eval_after, patch_cluster }
 *   targetKpi   — number (0-1), draws a horizontal target line
 */

const CHART_H = 120;
const BAR_W   = 48;
const GAP     = 24;

export default function MultiIterationChart({ iterations = [], targetKpi = 0.75 }) {
  if (!iterations.length) {
    return (
      <div className="card">
        <div className="section-title">Multi-Loop Progress</div>
        <p className="card-sub">No iteration data yet. Run Auto-Improve to populate this chart.</p>
      </div>
    );
  }

  // Build a flat list of data points: one "before" + one "after" per loop
  const points = [];
  iterations.forEach((loop, idx) => {
    if (idx === 0) {
      points.push({ label: `ckpt_v0.${idx}`, value: loop.eval_before, type: 'before' });
    }
    if (loop.eval_after != null) {
      points.push({
        label: `Loop ${loop.iteration_index}`,
        value: loop.eval_after,
        type: 'after',
        cluster: loop.patch_cluster,
      });
    }
  });

  const maxVal = 1.0;
  const svgW = points.length * (BAR_W + GAP) + GAP;

  function barY(val) {
    return CHART_H - Math.round((val / maxVal) * CHART_H);
  }
  function barH(val) {
    return Math.round((val / maxVal) * CHART_H);
  }

  const targetY = barY(targetKpi);

  return (
    <div className="card">
      <div className="section-title">Multi-Loop Progress</div>

      <div style={{ overflowX: 'auto', paddingBottom: 8 }}>
        <svg
          width={svgW}
          height={CHART_H + 48}
          style={{ display: 'block', minWidth: svgW }}
          aria-label="Multi-iteration success rate chart"
        >
          {/* Grid lines */}
          {[0.25, 0.5, 0.75, 1.0].map((v) => {
            const y = barY(v);
            return (
              <g key={v}>
                <line
                  x1={0} y1={y} x2={svgW} y2={y}
                  stroke="rgba(99,102,241,0.12)" strokeWidth={1} strokeDasharray="4 4"
                />
                <text x={4} y={y - 3} fill="var(--t3)" fontSize={9}>
                  {Math.round(v * 100)}%
                </text>
              </g>
            );
          })}

          {/* Target KPI line */}
          <line
            x1={0} y1={targetY} x2={svgW} y2={targetY}
            stroke="var(--green)" strokeWidth={1.5} strokeDasharray="6 3"
          />
          <text x={svgW - 42} y={targetY - 4} fill="var(--green)" fontSize={9} fontWeight="600">
            Target {Math.round(targetKpi * 100)}%
          </text>

          {/* Bars */}
          {points.map((pt, i) => {
            const x   = GAP + i * (BAR_W + GAP);
            const y   = barY(pt.value);
            const h   = barH(pt.value);
            const pct = Math.round(pt.value * 100);
            const fill = pt.type === 'before'
              ? 'rgba(99,102,241,0.4)'
              : pt.value >= targetKpi
                ? 'rgba(16,185,129,0.7)'
                : 'rgba(99,102,241,0.65)';
            return (
              <g key={i}>
                <rect
                  x={x} y={y} width={BAR_W} height={h}
                  rx={4}
                  fill={fill}
                  stroke={pt.type === 'before' ? 'rgba(99,102,241,0.5)' : 'rgba(16,185,129,0.4)'}
                  strokeWidth={1}
                />
                {/* Value label on top */}
                <text
                  x={x + BAR_W / 2} y={y - 4}
                  textAnchor="middle" fill="var(--t1)" fontSize={10} fontWeight="600"
                >
                  {pct}%
                </text>
                {/* X-axis label */}
                <text
                  x={x + BAR_W / 2} y={CHART_H + 14}
                  textAnchor="middle" fill="var(--t2)" fontSize={9}
                >
                  {pt.label}
                </text>
              </g>
            );
          })}

          {/* Connector lines between bars */}
          {points.map((pt, i) => {
            if (i === 0) return null;
            const x1 = GAP + (i - 1) * (BAR_W + GAP) + BAR_W;
            const y1 = barY(points[i - 1].value);
            const x2 = GAP + i * (BAR_W + GAP);
            const y2 = barY(pt.value);
            return (
              <line
                key={`conn-${i}`}
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="rgba(99,102,241,0.3)" strokeWidth={1} strokeDasharray="3 3"
              />
            );
          })}
        </svg>
      </div>

      {/* Legend row */}
      <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap' }}>
        {points
          .filter((p) => p.cluster)
          .map((p, i) => (
            <div key={i} className="card-sub" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: 'var(--a3)' }}>Loop {p.label.replace('Loop ', '')}</span>
              <span>·</span>
              <span>{p.cluster}</span>
              <span>·</span>
              <span style={{ color: 'var(--green)' }}>{Math.round(p.value * 100)}%</span>
            </div>
          ))}
      </div>
    </div>
  );
}
