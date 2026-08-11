import { ArrowLeftOutlined, DownloadOutlined, StopOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Col, Descriptions, Progress, Row, Space, Statistic, Typography, message } from 'antd';
import dayjs from 'dayjs';
import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import type { EvaluationRun } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { EvaluationResultVisuals } from '../components/EvaluationResultVisuals';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { StatusTag } from '../components/StatusTag';

function rate(value: unknown): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

export function EvaluationGroupDetailPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { groupId } = useParams();
  const group = useQuery({
    queryKey: ['evaluation-group', groupId],
    queryFn: () => api.evaluations.groups.get(groupId!),
    enabled: Boolean(groupId),
    refetchInterval: (query) => ['queued', 'running', 'starting', 'stopping'].includes(String(query.state.data?.status)) ? 3_000 : false,
  });
  const cancel = useMutation({
    mutationFn: () => api.evaluations.groups.cancel(groupId!),
    onSuccess: async () => {
      message.success(t('evaluation.groupCancelSubmitted'));
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['evaluation-group', groupId] }),
        queryClient.invalidateQueries({ queryKey: ['evaluation-groups'] }),
        queryClient.invalidateQueries({ queryKey: ['evaluations'] }),
      ]);
    },
    onError: (error) => message.error(error.message),
  });
  const value = group.data;
  const summary = value?.result?.summary ?? value?.result_summary ?? {};
  const queued = value?.runs.some((run) => run.status === 'queued');
  const result = value?.result;

  return <div className="page evaluation-group-detail-page">
    <PageIntro
      title={value?.name ?? t('evaluation.groupDetails')}
      subtitle={value ? `${t(`evaluation.kind.${value.kind}`, { defaultValue: value.kind })} · ${value.runs.length} ${t('evaluation.childRuns')}` : groupId}
      actions={<Space wrap>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/evaluations')}>{t('evaluation.title')}</Button>
        {result ? <Button icon={<DownloadOutlined />} href={api.evaluations.groups.resultDownloadUrl(groupId!)}>{t('evaluation.downloadResult')}</Button> : null}
        {queued ? <Button danger icon={<StopOutlined />} loading={cancel.isPending} onClick={() => cancel.mutate()}>{t('evaluation.cancelQueuedChildren')}</Button> : null}
      </Space>}
    />
    <AsyncState loading={group.isLoading} error={group.error} onRetry={() => void group.refetch()}>
      {value ? <Space direction="vertical" size="large" className="full-width">
        {value.error ? <Alert type="error" showIcon message={value.error} /> : null}
        <Row gutter={[16, 16]}>
          <Col xs={24} md={6}><Card><Statistic title={t('common.status')} valueRender={() => <StatusTag status={value.status} />} /></Card></Col>
          <Col xs={24} md={6}><Card><Statistic title={t('evaluation.successRate')} value={rate(summary.success_rate)} /></Card></Col>
          <Col xs={24} md={6}><Card><Statistic title={t('evaluation.completedRuns')} value={Number(summary.num_completed ?? value.runs.filter((run) => run.status === 'completed').length)} suffix={`/ ${value.runs.length}`} /></Card></Col>
          <Col xs={24} md={6}><Card><Statistic title={t('evaluation.completedEpisodes')} value={`${Number(summary.num_successes ?? 0)} / ${Number(summary.num_episodes ?? 0)}`} /></Card></Col>
        </Row>
        <Card title={t('evaluation.groupOverview')}>
          <Descriptions bordered column={{ xs: 1, md: 2 }}>
            <Descriptions.Item label={t('evaluation.kindLabel')}>{t(`evaluation.kind.${value.kind}`, { defaultValue: value.kind })}</Descriptions.Item>
            <Descriptions.Item label={t('common.owner')}>{value.owner_name ?? value.owner_id ?? '—'}</Descriptions.Item>
            <Descriptions.Item label={t('common.createdAt')}>{value.created_at ? dayjs(value.created_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
            <Descriptions.Item label={t('jobs.started')}>{value.started_at ? dayjs(value.started_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
            <Descriptions.Item label={t('jobs.finished')}>{value.finished_at ? dayjs(value.finished_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
            <Descriptions.Item label={t('evaluation.outputPath')}><Typography.Text code copyable>{value.result_path ?? '—'}</Typography.Text></Descriptions.Item>
          </Descriptions>
        </Card>
        <Card title={t('evaluation.childRuns')}>
          <Table<EvaluationRun>
            rowKey="id"
            pagination={false}
            dataSource={value.runs}
            onRow={(run) => ({ onClick: () => navigate(`/evaluations/${run.id}`) })}
            columns={[
              { title: '#', dataIndex: 'position', width: 64 },
              { title: t('evaluation.name'), dataIndex: 'name', render: (name: string, run) => <div><Typography.Text strong>{name}</Typography.Text><small className="table-subtitle">{run.evaluation_kind ?? 'standard'} · {run.suite || run.task_set || run.benchmark_id}</small></div> },
              { title: t('common.status'), dataIndex: 'status', render: (status) => <StatusTag status={status} /> },
              { title: t('evaluation.successRate'), render: (_, run) => run.result?.summary.success_rate == null ? '—' : <Progress percent={Number((run.result.summary.success_rate * 100).toFixed(1))} size="small" /> },
              { title: t('evaluation.gpuResources'), render: (_, run) => run.assigned_gpu_ids.length ? run.assigned_gpu_ids.map((id) => <span className="gpu-chip" key={id}>{id}</span>) : run.source_kind === 'managed_deployment' ? t('evaluation.reusedDeployment') : run.queue_position ? `#${run.queue_position}` : '—' },
              { title: t('common.actions'), render: (_, run) => <Button size="small" onClick={(event) => { event.stopPropagation(); navigate(`/evaluations/${run.id}`); }}>{t('common.view')}</Button> },
            ]}
          />
        </Card>
        {result ? <Card title={`${t('evaluation.results')} · ${result.schema_version}`}><EvaluationResultVisuals result={result} /></Card> : <Alert type="info" showIcon message={t('evaluation.waitingGroupResult')} />}
        <Card title={t('evaluation.configuration')}><pre className="code-block">{JSON.stringify(value.spec, null, 2)}</pre></Card>
      </Space> : null}
    </AsyncState>
  </div>;
}
