import { DownloadOutlined, FileOutlined } from '@ant-design/icons';
import { Button, Card, Empty, Progress, Space, Tag, Typography } from 'antd';
import ReactECharts from 'echarts-for-react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import type { EvaluationArtifact, EvaluationMatrix, EvaluationResult, EvaluationSeries } from '../api/types';
import { StatusTag } from './StatusTag';
import { ResizableTable as Table } from './ResizableTable';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function formatted(value: number | null, format: string): string {
  if (value == null) return '—';
  if (format === 'rate' || format === 'percent') return `${(value * 100).toFixed(1)}%`;
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function MatrixHeatmap({ matrix }: { matrix: EvaluationMatrix }) {
  const flat = matrix.values.flat().filter((value): value is number => typeof value === 'number');
  const min = flat.length ? Math.min(...flat) : 0;
  const max = flat.length ? Math.max(...flat) : 1;
  const color = (value: number | null) => {
    if (value == null) return 'transparent';
    const ratio = max === min ? 0.7 : Math.max(0, Math.min(1, (value - min) / (max - min)));
    return `color-mix(in srgb, var(--ant-color-success) ${Math.round(15 + ratio * 70)}%, transparent)`;
  };
  return (
    <Card size="small" title={matrix.name || matrix.id} extra={<Tag>{matrix.value_format}</Tag>}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', width: '100%', minWidth: 440 }}>
          <thead><tr><th style={{ padding: 8 }} />{matrix.column_labels.map((label) => <th key={label} style={{ padding: 8, textAlign: 'center' }}>{label}</th>)}</tr></thead>
          <tbody>{matrix.values.map((row, rowIndex) => <tr key={matrix.row_labels[rowIndex] ?? rowIndex}><th style={{ padding: 8, textAlign: 'left' }}>{matrix.row_labels[rowIndex] ?? rowIndex}</th>{row.map((value, columnIndex) => <td key={columnIndex} style={{ padding: 10, textAlign: 'center', border: '1px solid var(--ant-color-border-secondary)', background: color(value), fontVariantNumeric: 'tabular-nums' }}>{formatted(value, matrix.value_format)}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </Card>
  );
}

function SeriesChart({ series }: { series: EvaluationSeries }) {
  const categorical = series.points.some((point) => typeof point.x === 'string');
  const option = {
    tooltip: { trigger: 'axis' },
    grid: { left: 56, right: 24, top: 28, bottom: 48 },
    xAxis: categorical
      ? { type: 'category', name: series.x_label, data: series.points.map((point) => String(point.x)) }
      : { type: 'value', name: series.x_label },
    yAxis: { type: 'value', name: series.y_label, min: series.y_label.toLowerCase().includes('rate') ? 0 : undefined, max: series.y_label.toLowerCase().includes('rate') ? 1 : undefined },
    series: [{
      name: series.name,
      type: 'line',
      smooth: true,
      connectNulls: false,
      data: categorical
        ? series.points.map((point) => point.y)
        : series.points.map((point) => [point.x, point.y]),
    }],
  };
  return <Card size="small" title={series.name || series.id}><ReactECharts option={option} style={{ height: 300 }} notMerge lazyUpdate /></Card>;
}

function ArtifactCard({ artifact }: { artifact: EvaluationArtifact }) {
  const media = (artifact.media_type ?? '').toLowerCase();
  const path = artifact.path.toLowerCase();
  const video = artifact.kind === 'video' || media.startsWith('video/') || /\.(mp4|webm|mkv|avi)$/.test(path);
  const image = artifact.kind === 'image' || media.startsWith('image/') || /\.(png|jpe?g|gif|webp)$/.test(path);
  return (
    <Card size="small" title={artifact.name} extra={<Tag>{artifact.kind}</Tag>}>
      {video && artifact.url ? <video controls preload="metadata" src={artifact.url} style={{ width: '100%', maxHeight: 320 }} /> : null}
      {image && artifact.url ? <img src={artifact.url} alt={artifact.name} style={{ width: '100%', maxHeight: 320, objectFit: 'contain' }} /> : null}
      {!video && !image ? <Space direction="vertical"><FileOutlined /><Typography.Text code copyable>{artifact.path}</Typography.Text></Space> : null}
      {artifact.url ? <Button type="link" icon={<DownloadOutlined />} href={artifact.url} target="_blank">Download</Button> : null}
    </Card>
  );
}

export function EvaluationResultVisuals({ result, artifacts = [] }: { result: EvaluationResult; artifacts?: EvaluationArtifact[] }) {
  const { t, i18n } = useTranslation();
  const mergedArtifacts = Array.from(new Map([...result.artifacts, ...artifacts].map((item) => [item.id ?? item.path, item])).values());
  const rawCategories = isRecord(result.metadata.categories) ? result.metadata.categories : {};
  const categories = Object.entries(rawCategories).map(([name, raw]) => {
    const row = isRecord(raw) ? raw : {};
    const total = Number(row.total_count ?? row.total ?? 0);
    const successes = Number(row.success_count ?? row.successes ?? 0);
    return { name, total, successes, rate: total > 0 ? successes / total : 0 };
  });
  return (
    <Space direction="vertical" size="large" className="full-width">
      {categories.length ? <Card size="small" title={i18n.language === 'zh-CN' ? 'LIBERO-plus 类别汇总' : 'LIBERO-plus category summary'}><Table rowKey="name" pagination={false} dataSource={categories} columns={[{ title: i18n.language === 'zh-CN' ? '类别' : 'Category', dataIndex: 'name' }, { title: i18n.language === 'zh-CN' ? '成功数' : 'Successes', render: (_, row) => `${row.successes} / ${row.total}` }, { title: i18n.language === 'zh-CN' ? '成功率' : 'Success rate', dataIndex: 'rate', render: (value: number) => <Progress percent={Number((value * 100).toFixed(1))} size="small" /> }]} /></Card> : null}
      {result.matrices.map((matrix) => <MatrixHeatmap key={matrix.id} matrix={matrix} />)}
      {result.series.map((series) => <SeriesChart key={series.id} series={series} />)}
      {result.comparisons.length ? <Card size="small" title={i18n.language === 'zh-CN' ? '对比结果' : 'Comparisons'}><Table rowKey={(row) => String(row.id ?? row.name)} pagination={false} dataSource={result.comparisons} columns={[{ title: 'ID', render: (_, row) => String(row.name ?? row.id ?? '—') }, { title: i18n.language === 'zh-CN' ? '基线' : 'Baseline', render: (_, row) => row.baseline_success_rate == null ? '—' : formatted(Number(row.baseline_success_rate), 'rate') }, { title: i18n.language === 'zh-CN' ? '适应后' : 'Adapted', render: (_, row) => row.adapted_success_rate == null ? '—' : formatted(Number(row.adapted_success_rate), 'rate') }, { title: 'Δ', render: (_, row) => row.delta == null ? '—' : formatted(Number(row.delta), 'rate') }]} /></Card> : null}
      {result.children.length ? <Card size="small" title={i18n.language === 'zh-CN' ? '子评测' : 'Child evaluations'}><Table rowKey="evaluation_id" pagination={false} dataSource={result.children} columns={[{ title: i18n.language === 'zh-CN' ? '名称' : 'Name', render: (_, child) => <Link to={`/evaluations/${child.evaluation_id}`}>{child.name || child.evaluation_id}</Link> }, { title: i18n.language === 'zh-CN' ? '类型' : 'Kind', dataIndex: 'kind' }, { title: t('common.status'), dataIndex: 'status', render: (value) => <StatusTag status={value} /> }, { title: i18n.language === 'zh-CN' ? '成功率' : 'Success rate', render: (_, child) => child.summary.success_rate == null ? '—' : formatted(Number(child.summary.success_rate), 'rate') }]} /></Card> : null}
      {mergedArtifacts.length ? <div className="evaluation-video-grid">{mergedArtifacts.map((artifact) => <ArtifactCard key={artifact.id ?? artifact.path} artifact={artifact} />)}</div> : null}
      {!result.matrices.length && !result.series.length && !result.comparisons.length && !result.children.length && !mergedArtifacts.length && !categories.length ? <Empty description={i18n.language === 'zh-CN' ? '没有专项结果产物' : 'No specialized result artifacts'} /> : null}
    </Space>
  );
}
