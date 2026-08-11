import { useMemo } from 'react';
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { Empty, Segmented } from 'antd';
import { useState } from 'react';
import type { MetricPoint, ThemeMode } from '../api/types';

echarts.use([LineChart, GridComponent, TooltipComponent, CanvasRenderer]);

interface MetricChartProps {
  points: MetricPoint[];
  theme: ThemeMode;
}

export function MetricChart({ points, theme }: MetricChartProps) {
  const metricNames = useMemo(
    () => Array.from(new Set(points.flatMap((point) => Object.keys(point.metrics)))),
    [points],
  );
  const [chosen, setChosen] = useState<string>();
  const active = chosen && metricNames.includes(chosen) ? chosen : metricNames[0];

  const option = useMemo(() => ({
    animation: false,
    backgroundColor: 'transparent',
    textStyle: { color: theme === 'dark' ? '#c7d0e0' : '#334155' },
    tooltip: { trigger: 'axis' },
    grid: { left: 52, right: 24, top: 30, bottom: 44 },
    xAxis: {
      type: 'category',
      name: points.some((point) => point.step != null) ? 'step' : 'iteration',
      data: points.map((point, index) => point.step ?? point.iteration ?? index),
      axisLine: { lineStyle: { color: theme === 'dark' ? '#475569' : '#cbd5e1' } },
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: { lineStyle: { color: theme === 'dark' ? '#26334a' : '#eef2f7' } },
    },
    series: active ? [{
      name: active,
      data: points.map((point) => point.metrics[active] ?? null),
      type: 'line',
      smooth: true,
      showSymbol: false,
      connectNulls: false,
      lineStyle: { width: 2, color: '#5577e5' },
      areaStyle: { color: theme === 'dark' ? 'rgba(85,119,229,.12)' : 'rgba(85,119,229,.08)' },
    }] : [],
  }), [active, points, theme]);

  if (!metricNames.length) return <Empty />;
  return (
    <div className="metric-chart">
      <Segmented
        options={metricNames.map((name) => ({ label: name, value: name }))}
        value={active}
        onChange={(value) => setChosen(String(value))}
      />
      <ReactEChartsCore echarts={echarts} option={option} style={{ height: 360 }} notMerge lazyUpdate />
    </div>
  );
}
