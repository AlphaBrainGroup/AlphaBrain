import {
  ArrowLeftOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  CopyOutlined,
  DisconnectOutlined,
  LinkOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  StopOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Descriptions, Modal, Progress, Segmented, Space, Statistic, Switch, Tabs, Tag, Typography, message } from 'antd';
import dayjs from 'dayjs';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import type { Job } from '../api/types';
import { usePreferences } from '../app-context';
import { AsyncState } from '../components/AsyncState';
import { MetricChart } from '../components/MetricChart';
import { PageIntro } from '../components/PageIntro';
import { StatusTag } from '../components/StatusTag';
import { useJobStream } from '../hooks/useJobStream';

function elapsed(job?: Job): string {
  if (!job?.started_at) return '—';
  const end = job.finished_at ? dayjs(job.finished_at) : dayjs();
  const seconds = Math.max(0, end.diff(dayjs(job.started_at), 'second'));
  const hours = Math.floor(seconds / 3600);
  return `${hours.toString().padStart(2, '0')}:${Math.floor((seconds % 3600) / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
}

export function JobDetailPage() {
  const { t } = useTranslation();
  const { theme } = usePreferences();
  const { jobId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [follow, setFollow] = useState(true);
  const [metricSource, setMetricSource] = useState<'stream' | 'snapshot'>('stream');
  const terminalRef = useRef<HTMLDivElement>(null);
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const job = useQuery({ queryKey: ['job', jobId], queryFn: () => api.jobs.get(jobId!), enabled: Boolean(jobId), refetchInterval: 5_000 });
  const stream = useJobStream(jobId);
  const visibleStatus = stream.status ?? job.data?.status;

  useEffect(() => {
    if (follow && terminalRef.current) terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [follow, stream.logs]);

  const invalidate = async () => {
    await queryClient.invalidateQueries({ queryKey: ['job', jobId] });
    await queryClient.invalidateQueries({ queryKey: ['jobs'] });
  };
  const cancel = useMutation({ mutationFn: () => api.jobs.cancel(jobId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });
  const stop = useMutation({ mutationFn: () => api.jobs.stop(jobId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });
  const terminate = useMutation({ mutationFn: () => api.jobs.terminate(jobId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });
  const kill = useMutation({ mutationFn: () => api.jobs.forceKill(jobId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });

  const snapshotMetrics = useMemo(() => job.data?.latest_metrics ? [{ step: 0, metrics: job.data.latest_metrics }] : [], [job.data?.latest_metrics]);
  const metrics = metricSource === 'stream' && stream.metrics.length ? stream.metrics : snapshotMetrics;
  const canMutate = Boolean(
    me.data && (me.data.role === 'administrator' || (job.data?.owner_id && me.data.id === job.data.owner_id)),
  );
  const canStop = Boolean(canMutate && ['starting', 'running'].includes(visibleStatus ?? ''));
  const canCancel = Boolean(canMutate && (visibleStatus === 'queued' || visibleStatus === 'blocked'));

  return (
    <div className="page job-detail-page">
      <PageIntro
        title={job.data?.name ?? t('jobs.details')}
        subtitle={job.data?.stage_name ? `${t('jobs.stage')}: ${job.data.stage_name}` : jobId}
        actions={
          <Space>
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/jobs')}>{t('jobs.title')}</Button>
            {canCancel ? <Button danger icon={<CloseCircleOutlined />} loading={cancel.isPending} onClick={() => cancel.mutate()}>{t('jobs.cancelQueue')}</Button> : null}
            {canStop ? <Button danger icon={<PauseCircleOutlined />} loading={stop.isPending} onClick={() => stop.mutate()}>{t('jobs.stop')}</Button> : null}
            {canMutate && visibleStatus === 'stopping' ? <Button danger icon={<StopOutlined />} loading={terminate.isPending} onClick={() => terminate.mutate()}>{t('jobs.terminate')}</Button> : null}
            {(canStop || visibleStatus === 'stopping') && me.data?.role === 'administrator' ? <Button type="primary" danger icon={<StopOutlined />} onClick={() => Modal.confirm({ title: t('jobs.forceKill'), content: t('jobs.forceConfirm'), okButtonProps: { danger: true }, onOk: () => kill.mutate() })}>{t('jobs.forceKill')}</Button> : null}
          </Space>
        }
      />
      <AsyncState loading={job.isLoading} error={job.error} onRetry={() => void job.refetch()}>
        <div className="job-summary">
          <Card><Statistic title={t('common.status')} valueRender={() => <StatusTag status={visibleStatus} />} prefix={visibleStatus === 'running' ? <ReloadOutlined spin /> : <CheckCircleOutlined />} /></Card>
          <Card><Statistic title={t('jobs.duration')} value={elapsed(job.data)} prefix={<ClockCircleOutlined />} /></Card>
          <Card><Statistic title={t('jobs.gpus')} value={(job.data?.gpu_ids ?? []).join(', ') || '—'} prefix={<LinkOutlined />} /></Card>
          <Card><Statistic title={t('jobs.queue')} value={job.data?.queue_position ? `#${job.data.queue_position}` : '—'} /></Card>
        </div>
        {job.data?.error_summary ? <Alert className="detail-alert" type="error" showIcon message={job.data.error_summary} /> : null}
        <Card className="job-tabs-card">
          <Tabs
            defaultActiveKey="overview"
            items={[
              {
                key: 'overview', label: t('jobs.overview'),
                children: (
                  <div className="overview-grid">
                    <Descriptions bordered column={{ xs: 1, md: 2 }}>
                      <Descriptions.Item label={t('common.owner')}>{job.data?.owner_name ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('jobs.stage')}>{job.data?.stage_name ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label="PID">{job.data?.pid ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('common.createdAt')}>{job.data?.created_at ? dayjs(job.data.created_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('jobs.started')}>{job.data?.started_at ? dayjs(job.data.started_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('jobs.finished')}>{job.data?.finished_at ? dayjs(job.data.finished_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('checkpoints.experiment')} span={2}>{job.data?.experiment_id ? <Link to={`/experiments/${job.data.experiment_id}`}>{job.data.experiment_id}</Link> : '—'}</Descriptions.Item>
                    </Descriptions>
                    {job.data?.progress != null ? <Card size="small" title={t('jobs.progress')}><Progress percent={Math.round(job.data.progress <= 1 ? job.data.progress * 100 : job.data.progress)} status={visibleStatus === 'failed' ? 'exception' : visibleStatus === 'completed' ? 'success' : 'active'} /></Card> : null}
                    {job.data?.command ? (
                      <div><Space><Typography.Text strong>{t('jobs.command')}</Typography.Text><Button type="text" size="small" icon={<CopyOutlined />} onClick={() => { void navigator.clipboard.writeText(job.data?.command ?? ''); message.success(t('common.copied')); }} /></Space><pre className="command-preview">{job.data.command}</pre></div>
                    ) : null}
                  </div>
                ),
              },
              {
                key: 'logs', label: t('jobs.logs'),
                children: (
                  <div>
                    <div className="terminal-toolbar">
                      <Tag
                        icon={stream.connected || stream.ended ? <CheckCircleOutlined /> : <DisconnectOutlined />}
                        color={stream.ended ? 'default' : stream.connected ? 'green' : 'orange'}
                      >
                        {stream.ended ? t('jobs.streamEnded') : stream.connected ? t('jobs.streamConnected') : t('jobs.streamDisconnected')}
                      </Tag>
                      <Space><span>{t('jobs.follow')}</span><Switch size="small" checked={follow} onChange={setFollow} /></Space>
                    </div>
                    <div className="terminal" ref={terminalRef}>{stream.logs.length ? stream.logs.map((line, index) => <div key={`${index}-${line.slice(0, 12)}`}><span className="line-no">{String(index + 1).padStart(4, '0')}</span>{line}</div>) : <span className="terminal-empty">{t('jobs.noLogs')}</span>}</div>
                  </div>
                ),
              },
              {
                key: 'metrics', label: t('jobs.metrics'),
                children: (
                  <div>
                    <div className="chart-toolbar"><Segmented value={metricSource} onChange={(value) => setMetricSource(value as 'stream' | 'snapshot')} options={[{ label: t('jobs.liveStream'), value: 'stream' }, { label: t('jobs.latestSnapshot'), value: 'snapshot' }]} /></div>
                    {metrics.length ? <MetricChart points={metrics} theme={theme} /> : <Alert type="info" showIcon message={t('jobs.noMetrics')} />}
                  </div>
                ),
              },
            ]}
          />
        </Card>
      </AsyncState>
    </div>
  );
}
