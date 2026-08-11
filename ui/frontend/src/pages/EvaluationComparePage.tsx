import { ArrowLeftOutlined, BarChartOutlined, TrophyOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Alert, Button, Card, Progress, Statistic, Tag, Typography } from 'antd';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import type { EvaluationComparisonItem } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';

function percent(value?: number): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

export function EvaluationComparePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const ids = useMemo(() => Array.from(new Set((searchParams.get('ids') ?? '').split(',').filter(Boolean))).slice(0, 8), [searchParams]);
  const comparison = useQuery({ queryKey: ['evaluation-comparison', ids], queryFn: () => api.evaluations.compare(ids), enabled: ids.length >= 2, retry: false });
  const taskRows = useMemo(() => {
    const taskIds = Array.from(new Set((comparison.data?.items ?? []).flatMap((item) => item.result?.tasks.map((task) => task.id) ?? [])));
    return taskIds.map((taskId) => ({ taskId, values: Object.fromEntries((comparison.data?.items ?? []).map((item) => [item.evaluation.id, item.result?.tasks.find((task) => task.id === taskId)?.success_rate])) }));
  }, [comparison.data?.items]);

  return <div className="page evaluation-compare-page">
    <PageIntro title={t('evaluation.compareTitle')} subtitle={t('evaluation.compareSubtitle')} actions={<Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/evaluations')}>{t('evaluation.title')}</Button>} />
    {ids.length < 2 ? <Alert type="warning" showIcon message={t('evaluation.compareIssue.count')} action={<Button size="small" onClick={() => navigate('/evaluations')}>{t('evaluation.selectRuns')}</Button>} /> : <AsyncState loading={comparison.isLoading} error={comparison.error} onRetry={() => void comparison.refetch()}>
      {comparison.data?.compatible === false ? <Alert className="comparison-hint" type="error" showIcon message={t('evaluation.compareIncompatible')} /> : null}
      {comparison.data?.warnings.map((warning) => <Alert key={warning} className="comparison-hint" type="warning" showIcon message={t(`evaluation.compareWarning.${warning}`)} />)}
      <div className="evaluation-compare-summary">{comparison.data?.items.map((item) => <CompareCard key={item.evaluation.id} item={item} onOpen={() => navigate(`/evaluations/${item.evaluation.id}`)} />)}</div>
      <Card title={t('evaluation.taskComparison')} className="evaluation-compare-table"><Table rowKey="taskId" pagination={false} scroll={{ x: 720 }} dataSource={taskRows} columns={[
        { title: t('evaluation.task'), dataIndex: 'taskId', fixed: 'left', width: 220, render: (value: string) => <Typography.Text strong>{value}</Typography.Text> },
        ...(comparison.data?.items ?? []).map((item) => ({ title: item.evaluation.name, key: item.evaluation.id, width: 190, render: (_: unknown, row: { values: Record<string, number | undefined> }) => { const value = row.values[item.evaluation.id]; return value == null ? '—' : <div className="evaluation-compare-rate"><Progress type="circle" size={48} percent={Math.round(value * 100)} /><Typography.Text>{percent(value)}</Typography.Text></div>; } })),
      ]} /></Card>
    </AsyncState>}
  </div>;
}

function CompareCard({ item, onOpen }: { item: EvaluationComparisonItem; onOpen: () => void }) {
  const { t } = useTranslation();
  const result = item.result ?? item.evaluation.result;
  return <Card hoverable onClick={onOpen} title={item.evaluation.name} extra={<Tag>{item.evaluation.preset}</Tag>}><Statistic title={t('evaluation.successRate')} value={percent(result?.summary.success_rate)} prefix={<TrophyOutlined />} /><div className="evaluation-compare-meta"><span><BarChartOutlined /> {result ? `${result.summary.num_successes}/${result.summary.num_episodes}` : '—'}</span><span>{item.evaluation.suite ?? item.evaluation.task_set ?? item.evaluation.split ?? item.evaluation.benchmark_id}</span></div></Card>;
}
