import {
  ArrowLeftOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  DisconnectOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  PlayCircleOutlined,
  StopOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Modal,
  Progress,
  Space,
  Statistic,
  Switch,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd';
import dayjs from 'dayjs';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import type { EvaluationEpisodeMetric, EvaluationTaskMetric, EvaluationVideo } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { EvaluationResultVisuals } from '../components/EvaluationResultVisuals';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { StatusTag } from '../components/StatusTag';
import { useEvaluationStream } from '../hooks/useEvaluationStream';

function duration(seconds?: number): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = Math.floor(seconds % 60);
  return hours ? `${hours}h ${minutes}m ${rest}s` : minutes ? `${minutes}m ${rest}s` : `${rest}s`;
}

function elapsed(start?: string, end?: string): string {
  if (!start) return '—';
  return duration(Math.max(0, (end ? dayjs(end) : dayjs()).diff(dayjs(start), 'second')));
}

function percent(value?: number): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function VideoGallery({ videos, emptyText, successText, failureText }: { videos: EvaluationVideo[]; emptyText: string; successText: string; failureText: string }) {
  if (!videos.length) return <div className="evaluation-empty-artifact"><PlayCircleOutlined /><Typography.Text type="secondary">{emptyText}</Typography.Text></div>;
  return <div className="evaluation-video-grid">{videos.map((video) => <Card key={video.id || video.url} size="small" title={video.name} extra={video.url ? <Button type="link" size="small" icon={<DownloadOutlined />} href={video.url} download>{video.task_name ?? video.task_id ?? ''}</Button> : null}>{video.url ? <video controls preload="metadata" src={video.url}>{emptyText}</video> : <Alert type="warning" showIcon message={video.path ?? emptyText} />}<Space wrap className="evaluation-video-meta">{video.task_name || video.task_id ? <Tag>{video.task_name ?? video.task_id}</Tag> : null}{video.episode_index != null ? <Tag>Episode {video.episode_index}</Tag> : null}{video.success != null ? <Tag color={video.success ? 'green' : 'red'}>{video.success ? successText : failureText}</Tag> : null}</Space></Card>)}</div>;
}

export function EvaluationDetailPage() {
  const { t } = useTranslation();
  const { evaluationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const terminalRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const evaluation = useQuery({ queryKey: ['evaluation', evaluationId], queryFn: () => api.evaluations.get(evaluationId!), enabled: Boolean(evaluationId), refetchInterval: 3_000 });
  const stream = useEvaluationStream(evaluationId);
  const visibleStatus = stream.status ?? evaluation.data?.status;
  const resultQuery = useQuery({ queryKey: ['evaluation-result', evaluationId], queryFn: () => api.evaluations.result(evaluationId!), enabled: Boolean(evaluationId && visibleStatus === 'completed' && !evaluation.data?.result), retry: false });
  const artifactQuery = useQuery({ queryKey: ['evaluation-artifacts', evaluationId], queryFn: () => api.evaluations.artifacts(evaluationId!), enabled: Boolean(evaluationId && ['running', 'completed', 'failed', 'stopped', 'cancelled', 'interrupted'].includes(visibleStatus ?? '')), retry: false, refetchInterval: visibleStatus === 'running' ? 10_000 : false });
  const result = resultQuery.data ?? evaluation.data?.result;
  const videos = useMemo(() => {
    const artifactVideos: EvaluationVideo[] = (artifactQuery.data ?? [])
      .filter((artifact) => artifact.kind === 'video' || artifact.media_type?.startsWith('video/') || /\.(mp4|webm|mkv|avi)$/i.test(artifact.path))
      .map((artifact) => ({ id: artifact.id ?? artifact.path, name: artifact.name, path: artifact.path, url: artifact.url }));
    const merged = [...(result?.videos ?? []), ...(evaluation.data?.videos ?? []), ...artifactVideos];
    return Array.from(new Map(merged.map((video) => [video.path || video.id || video.url, video])).values());
  }, [artifactQuery.data, evaluation.data?.videos, result?.videos]);
  const progressValue = stream.progress ?? evaluation.data?.progress;
  const progress = progressValue == null ? undefined : Math.max(0, Math.min(100, progressValue <= 1 ? progressValue * 100 : progressValue));
  const canMutate = Boolean(me.data && (me.data.role === 'administrator' || me.data.deployment_mode === 'personal' || (evaluation.data?.owner_id && evaluation.data.owner_id === me.data.id)));

  useEffect(() => {
    if (follow && terminalRef.current) terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [follow, stream.logs]);

  const invalidate = async () => {
    await queryClient.invalidateQueries({ queryKey: ['evaluation', evaluationId] });
    await queryClient.invalidateQueries({ queryKey: ['evaluations'] });
  };
  const cancel = useMutation({ mutationFn: () => api.evaluations.cancel(evaluationId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });
  const stop = useMutation({ mutationFn: () => api.evaluations.stop(evaluationId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });
  const forceKill = useMutation({ mutationFn: () => api.evaluations.forceKill(evaluationId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });
  const retryWandb = useMutation({ mutationFn: () => api.evaluations.retryWandb(evaluationId!), onSuccess: async () => { await invalidate(); message.success(t('evaluation.wandbRetrySubmitted')); }, onError: (error) => message.error(error.message) });
  const canCancel = canMutate && visibleStatus === 'queued';
  const canStop = canMutate && ['starting', 'running'].includes(visibleStatus ?? '');
  const canForceKill = me.data?.role === 'administrator' && ['starting', 'running', 'stopping'].includes(visibleStatus ?? '');

  return <div className="page evaluation-detail-page">
    <PageIntro title={evaluation.data?.name ?? t('evaluation.details')} subtitle={evaluation.data ? `${evaluation.data.evaluation_kind} · ${evaluation.data.benchmark_id} · ${evaluation.data.combination_id}` : evaluationId} actions={<Space wrap><Button icon={<ArrowLeftOutlined />} onClick={() => navigate(evaluation.data?.group_id ? `/evaluation-groups/${evaluation.data.group_id}` : '/evaluations')}>{evaluation.data?.group_id ? 'Group' : t('evaluation.title')}</Button>{result ? <Button icon={<BarChartOutlined />} onClick={() => navigate(`/evaluations/compare?ids=${evaluationId}`)} disabled>{t('evaluation.compare')}</Button> : null}{canCancel ? <Button danger icon={<CloseCircleOutlined />} loading={cancel.isPending} onClick={() => cancel.mutate()}>{t('evaluation.cancelQueue')}</Button> : null}{canStop ? <Button danger icon={<StopOutlined />} loading={stop.isPending} onClick={() => stop.mutate()}>{t('evaluation.stop')}</Button> : null}{canForceKill ? <Button type="primary" danger icon={<StopOutlined />} loading={forceKill.isPending} onClick={() => Modal.confirm({ title: t('evaluation.forceKill'), content: t('evaluation.forceKillConfirm'), okButtonProps: { danger: true }, onOk: () => forceKill.mutateAsync() })}>{t('evaluation.forceKill')}</Button> : null}</Space>} />
    <AsyncState loading={evaluation.isLoading || me.isLoading} error={evaluation.error ?? me.error} onRetry={() => { void evaluation.refetch(); void me.refetch(); }}>
      <div className="job-summary"><Card><Statistic title={t('common.status')} valueRender={() => <StatusTag status={visibleStatus} />} prefix={visibleStatus === 'completed' ? <CheckCircleOutlined /> : <ExperimentOutlined />} /></Card><Card><Statistic title={t('evaluation.successRate')} value={percent(result?.summary.success_rate)} prefix={<BarChartOutlined />} /></Card><Card><Statistic title={t('evaluation.completedEpisodes')} value={result ? `${result.summary.num_successes} / ${result.summary.num_episodes}` : '—'} /></Card><Card><Statistic title={t('jobs.duration')} value={result?.duration_seconds != null ? duration(result.duration_seconds) : elapsed(evaluation.data?.started_at, evaluation.data?.finished_at)} prefix={<ClockCircleOutlined />} /></Card></div>
      {progress != null && ['starting', 'running', 'stopping'].includes(visibleStatus ?? '') ? <Card size="small" className="evaluation-progress"><Space direction="vertical" className="full-width"><Space><Typography.Text strong>{stream.phase ?? evaluation.data?.phase ?? t('evaluation.runningPhase')}</Typography.Text><Typography.Text type="secondary">{progress.toFixed(0)}%</Typography.Text></Space><Progress percent={progress} status={visibleStatus === 'stopping' ? 'exception' : 'active'} /></Space></Card> : null}
      {evaluation.data?.error_summary ? <Alert className="detail-alert" type="error" showIcon message={evaluation.data.error_summary} /> : null}
      {evaluation.data?.wandb.enabled && evaluation.data.wandb_status === 'failed' ? <Alert className="detail-alert" type="warning" showIcon message={t('evaluation.wandbUploadFailed')} description={evaluation.data.wandb_error} action={<Button size="small" loading={retryWandb.isPending} onClick={() => retryWandb.mutate()}>{t('evaluation.retryWandb')}</Button>} /> : null}
      {evaluation.data?.wandb_run_url ? <Alert className="detail-alert" type="success" showIcon message={t('evaluation.wandbUploaded')} action={<Button type="link" href={evaluation.data.wandb_run_url} target="_blank">W&B</Button>} /> : null}
      <Card className="job-tabs-card"><Tabs defaultActiveKey={result ? 'results' : 'overview'} items={[
        { key: 'overview', label: t('jobs.overview'), children: <Descriptions bordered column={{ xs: 1, md: 2 }}><Descriptions.Item label={t('common.owner')}>{evaluation.data?.owner_name ?? '—'}</Descriptions.Item><Descriptions.Item label="PID">{evaluation.data?.pid ?? '—'}</Descriptions.Item><Descriptions.Item label="Kind">{evaluation.data?.evaluation_kind ?? 'standard'}</Descriptions.Item><Descriptions.Item label="Result schema">{evaluation.data?.result_schema_version ?? result?.schema_version ?? '—'}</Descriptions.Item><Descriptions.Item label={t('evaluation.benchmark')}>{evaluation.data?.benchmark_id ?? '—'}</Descriptions.Item><Descriptions.Item label={t('evaluation.scale')}>{evaluation.data?.preset ? t(`evaluation.preset.${evaluation.data.preset}`) : '—'}</Descriptions.Item><Descriptions.Item label={t('evaluation.suite')}>{evaluation.data?.suite ?? '—'}</Descriptions.Item><Descriptions.Item label={t('evaluation.taskSet')}>{evaluation.data?.task_set ?? '—'}</Descriptions.Item><Descriptions.Item label={t('evaluation.split')}>{evaluation.data?.split ?? '—'}</Descriptions.Item><Descriptions.Item label={t('builder.gpuIds')}>{evaluation.data?.assigned_gpu_ids.join(', ') || (evaluation.data?.queue_position ? `#${evaluation.data.queue_position}` : '—')}</Descriptions.Item><Descriptions.Item label="Backbone">{evaluation.data?.backbone_id ?? '—'}</Descriptions.Item><Descriptions.Item label="Action Head">{evaluation.data?.action_head_id ?? '—'}</Descriptions.Item><Descriptions.Item label={t('evaluation.checkpoint')} span={2}><Typography.Text code>{evaluation.data?.checkpoint_path ?? evaluation.data?.checkpoint_id ?? '—'}</Typography.Text></Descriptions.Item><Descriptions.Item label={t('common.createdAt')}>{evaluation.data?.created_at ? dayjs(evaluation.data.created_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item><Descriptions.Item label={t('jobs.started')}>{evaluation.data?.started_at ? dayjs(evaluation.data.started_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item><Descriptions.Item label={t('jobs.finished')}>{evaluation.data?.finished_at ? dayjs(evaluation.data.finished_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item><Descriptions.Item label={t('evaluation.outputPath')}><Typography.Text code>{evaluation.data?.output_path ?? '—'}</Typography.Text></Descriptions.Item></Descriptions> },
        { key: 'results', label: t('evaluation.results'), children: result ? <div className="evaluation-results"><div className="evaluation-result-head"><div><Typography.Title level={4}>{t('evaluation.taskMetrics')}</Typography.Title><Typography.Text type="secondary">{result.schema_version} · {result.kind ?? evaluation.data?.evaluation_kind}</Typography.Text></div><Button icon={<DownloadOutlined />} href={api.evaluations.resultDownloadUrl(evaluationId!)}>{t('evaluation.downloadResult')}</Button></div>{result.tasks.length ? <Table<EvaluationTaskMetric> rowKey="id" pagination={false} dataSource={result.tasks} columns={[{ title: t('evaluation.task'), render: (_, task) => <div><Typography.Text strong>{task.name ?? task.id}</Typography.Text><small className="table-subtitle">{task.suite}</small></div> }, { title: t('evaluation.successes'), render: (_, task) => `${task.num_successes} / ${task.num_episodes}` }, { title: t('evaluation.successRate'), dataIndex: 'success_rate', width: 280, render: (value: number) => <div className="evaluation-rate-cell"><Progress percent={Number((value * 100).toFixed(1))} size="small" /><Typography.Text strong>{percent(value)}</Typography.Text></div> }, { title: t('jobs.duration'), dataIndex: 'duration_seconds', render: duration }]} /> : null}{result.episodes.length ? <><Typography.Title level={4} className="evaluation-episode-title">{t('evaluation.episodes')}</Typography.Title><Table<EvaluationEpisodeMetric> size="small" rowKey="id" dataSource={result.episodes} pagination={{ pageSize: 20 }} columns={[{ title: 'Episode', dataIndex: 'episode_index' }, { title: t('evaluation.task'), dataIndex: 'task_id' }, { title: 'Seed', dataIndex: 'seed' }, { title: t('evaluation.result'), dataIndex: 'success', render: (value: boolean) => <Tag color={value ? 'green' : 'red'}>{value ? t('evaluation.success') : t('evaluation.failure')}</Tag> }, { title: t('evaluation.steps'), dataIndex: 'steps' }, { title: t('jobs.duration'), dataIndex: 'duration_seconds', render: duration }]} /></> : null}<EvaluationResultVisuals result={result} artifacts={artifactQuery.data ?? []} /></div> : <Alert type={visibleStatus === 'failed' ? 'error' : 'info'} showIcon message={visibleStatus === 'failed' ? t('evaluation.noResultOnFailure') : t('evaluation.waitingResult')} /> },
        { key: 'videos', label: `${t('evaluation.videos')} (${videos.length})`, children: <VideoGallery videos={videos} emptyText={t('evaluation.noVideos')} successText={t('evaluation.success')} failureText={t('evaluation.failure')} /> },
        { key: 'logs', label: t('jobs.logs'), children: <div><div className="terminal-toolbar"><Tag icon={stream.connected || stream.ended ? <CheckCircleOutlined /> : <DisconnectOutlined />} color={stream.ended ? 'default' : stream.connected ? 'green' : 'orange'}>{stream.ended ? t('jobs.streamEnded') : stream.connected ? t('jobs.streamConnected') : t('jobs.streamDisconnected')}</Tag><Space><span>{t('jobs.follow')}</span><Switch size="small" checked={follow} onChange={setFollow} /></Space></div><div className="terminal" ref={terminalRef}>{stream.logs.length ? stream.logs.map((line, index) => <div key={`${index}-${line.slice(0, 16)}`}><span className="line-no">{String(index + 1).padStart(4, '0')}</span>{line}</div>) : <span className="terminal-empty">{t('jobs.noLogs')}</span>}</div></div> },
        { key: 'config', label: t('evaluation.configuration'), children: <div className="overview-grid"><pre className="config-preview large">{JSON.stringify({ benchmark_id: evaluation.data?.benchmark_id, preset: evaluation.data?.preset, suite: evaluation.data?.suite, task_set: evaluation.data?.task_set, split: evaluation.data?.split, parameters: evaluation.data?.parameters, wandb: evaluation.data?.wandb, wandb_status: evaluation.data?.wandb_status }, null, 2)}</pre></div> },
      ]} /></Card>
    </AsyncState>
  </div>;
}
