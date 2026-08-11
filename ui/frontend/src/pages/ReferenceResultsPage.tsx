import { LinkOutlined, ReloadOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Card, Empty, Select, Space, Tag, Typography } from 'antd';
import ReactECharts from 'echarts-for-react';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api } from '../api/client';
import type { ReferenceMetricDefinition, ReferenceResultRow, ReferenceResultSet } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';

function metricValue(value: number | null | undefined, metric: ReferenceMetricDefinition): string {
  if (value == null) return '—';
  if (metric.unit === 'percent') return `${value.toFixed(Number.isInteger(value) ? 0 : 1)}%`;
  if (metric.unit === 'seconds') return `${value}s`;
  return String(value);
}

function signatureTags(value: Record<string, string | number | boolean>) {
  return Object.entries(value).map(([key, item]) => <Tag key={key}>{key}={String(item)}</Tag>);
}

function ReferenceTable({ result }: { result: ReferenceResultSet }) {
  const { t } = useTranslation();
  return <Table<ReferenceResultRow>
    rowKey="id"
    size="small"
    scroll={{ x: 720 }}
    pagination={false}
    dataSource={result.rows}
    columns={[
      { title: t('referenceResults.model'), render: (_, row) => <Space direction="vertical" size={0}><Typography.Text strong>{row.label}</Typography.Text><Typography.Text type="secondary">{row.origin}</Typography.Text></Space> },
      ...result.metrics.map((metric) => ({
        title: metric.label,
        key: metric.id,
        align: 'right' as const,
        render: (_: unknown, row: ReferenceResultRow) => metricValue(row.values[metric.id], metric),
      })),
      { title: t('referenceResults.context'), render: (_: unknown, row: ReferenceResultRow) => <Space wrap>{signatureTags(row.context)}</Space> },
    ]}
  />;
}

function ReferenceSeries({ result }: { result: ReferenceResultSet }) {
  const percentOnly = result.metrics.every((metric) => metric.unit === 'percent');
  return <ReactECharts
    style={{ height: 360 }}
    option={{
      tooltip: { trigger: 'axis' },
      legend: { type: 'scroll', bottom: 0 },
      grid: { top: 24, left: 55, right: 24, bottom: 64 },
      xAxis: { type: 'value', name: 'step' },
      yAxis: { type: 'value', min: percentOnly ? 0 : undefined, max: percentOnly ? 100 : undefined },
      series: result.series.flatMap((series) => result.metrics.map((metric) => ({
        name: `${series.label} · ${metric.label}`,
        type: 'line',
        showSymbol: false,
        data: series.points
          .filter((point) => point.values[metric.id] != null)
          .map((point) => [point.step, point.values[metric.id]]),
      }))),
    }}
  />;
}

export function ReferenceResultsPage() {
  const { t } = useTranslation();
  const query = useQuery({ queryKey: ['reference-results'], queryFn: api.referenceResults.list });
  const [benchmark, setBenchmark] = useState('all');
  const [group, setGroup] = useState('all');
  const benchmarks = useMemo(() => Array.from(new Set((query.data?.result_sets ?? []).map((item) => item.benchmark_id))).sort(), [query.data]);
  const groups = useMemo(() => Array.from(new Set((query.data?.result_sets ?? []).map((item) => item.group))).sort(), [query.data]);
  const items = useMemo(() => (query.data?.result_sets ?? []).filter((item) =>
    (benchmark === 'all' || item.benchmark_id === benchmark) && (group === 'all' || item.group === group)), [benchmark, group, query.data]);

  return (
    <div className="page reference-results-page">
      <PageIntro
        title={t('referenceResults.title')}
        subtitle={t('referenceResults.subtitle')}
        actions={<Button icon={<ReloadOutlined />} onClick={() => void query.refetch()}>{t('common.refresh')}</Button>}
      />
      <AsyncState loading={query.isLoading} error={query.error} empty={!query.data?.result_sets.length} onRetry={() => void query.refetch()}>
        {query.data ? (
          <>
            <Alert
              showIcon
              type="info"
              message={t('referenceResults.snapshotTitle', { date: query.data.captured_at.slice(0, 10), version: query.data.version })}
              description={query.data.disclaimer}
              action={<Button icon={<LinkOutlined />} href={query.data.source_url} target="_blank" rel="noreferrer">{t('referenceResults.source')}</Button>}
            />
            <Card>
              <Space wrap>
                <Select value={benchmark} onChange={setBenchmark} style={{ minWidth: 220 }} options={[{ value: 'all', label: t('referenceResults.allBenchmarks') }, ...benchmarks.map((value) => ({ value, label: value }))]} />
                <Select value={group} onChange={setGroup} style={{ minWidth: 220 }} options={[{ value: 'all', label: t('referenceResults.allGroups') }, ...groups.map((value) => ({ value, label: value }))]} />
              </Space>
            </Card>
            {!items.length ? <Card><Empty description={t('common.noData')} /></Card> : items.map((result) => (
              <Card
                key={result.id}
                title={<Space wrap><Typography.Text strong>{result.title}</Typography.Text><Tag>{result.benchmark_id}</Tag><Tag>{result.category}</Tag></Space>}
              >
                <Space wrap style={{ marginBottom: 12 }}>{signatureTags(result.signature)}</Space>
                {result.kind === 'table' ? <ReferenceTable result={result} /> : <ReferenceSeries result={result} />}
                <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>{result.notes}</Typography.Paragraph>
              </Card>
            ))}
          </>
        ) : null}
      </AsyncState>
    </div>
  );
}
