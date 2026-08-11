import {
  ArrowLeftOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  CopyOutlined,
  DisconnectOutlined,
  KeyOutlined,
  LinkOutlined,
  ReloadOutlined,
  StopOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Descriptions, Modal, Space, Statistic, Switch, Tabs, Tag, Typography, message } from 'antd';
import dayjs from 'dayjs';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import type { DeploymentMutationResult, ModelDeployment } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { DeploymentApiKeyModal } from '../components/DeploymentApiKeyModal';
import { PageIntro } from '../components/PageIntro';
import { StatusTag } from '../components/StatusTag';
import { useDeploymentStream } from '../hooks/useDeploymentStream';
import { deploymentEndpoint, pythonClientSnippet } from './deployment-form';

function elapsed(deployment?: ModelDeployment): string {
  if (!deployment?.started_at) return '—';
  const end = deployment.finished_at ? dayjs(deployment.finished_at) : dayjs();
  const seconds = Math.max(0, end.diff(dayjs(deployment.started_at), 'second'));
  const hours = Math.floor(seconds / 3600);
  return `${String(hours).padStart(2, '0')}:${String(Math.floor((seconds % 3600) / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

export function DeploymentDetailPage() {
  const { t } = useTranslation();
  const { deploymentId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const terminalRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  const [secret, setSecret] = useState<DeploymentMutationResult>();
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const deployment = useQuery({
    queryKey: ['deployment', deploymentId],
    queryFn: () => api.deployments.get(deploymentId!),
    enabled: Boolean(deploymentId),
    refetchInterval: 3_000,
  });
  const stream = useDeploymentStream(deploymentId);
  const visibleStatus = stream.status ?? deployment.data?.status;
  const endpoint = deployment.data ? deploymentEndpoint(deployment.data) : undefined;
  const canMutate = Boolean(me.data && (
    me.data.role === 'administrator'
    || (deployment.data?.owner_id && deployment.data.owner_id === me.data.id)
    || me.data.deployment_mode === 'personal'
  ));

  useEffect(() => {
    if (follow && terminalRef.current) terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [follow, stream.logs]);

  const invalidate = async () => {
    await queryClient.invalidateQueries({ queryKey: ['deployment', deploymentId] });
    await queryClient.invalidateQueries({ queryKey: ['deployments'] });
  };
  const cancel = useMutation({ mutationFn: () => api.deployments.cancel(deploymentId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });
  const stop = useMutation({ mutationFn: () => api.deployments.stop(deploymentId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });
  const restart = useMutation({
    mutationFn: () => api.deployments.restart(deploymentId!),
    onSuccess: async (result) => {
      await invalidate();
      message.success(t('deployment.restartSubmitted'));
      if (result.api_key) setSecret(result);
    },
    onError: (error) => message.error(error.message),
  });
  const rotate = useMutation({
    mutationFn: () => api.deployments.rotateKey(deploymentId!),
    onSuccess: async (result) => {
      await invalidate();
      if (result.api_key) setSecret(result);
    },
    onError: (error) => message.error(error.message),
  });
  const forceKill = useMutation({ mutationFn: () => api.deployments.forceKill(deploymentId!), onSuccess: invalidate, onError: (error) => message.error(error.message) });

  const copy = (value: string) => {
    void navigator.clipboard.writeText(value);
    message.success(t('common.copied'));
  };
  const canCancel = canMutate && visibleStatus === 'queued';
  const canStop = canMutate && ['starting', 'running'].includes(visibleStatus ?? '');
  const canRestart = canMutate && ['stopped', 'failed', 'cancelled'].includes(visibleStatus ?? '');
  const canRotate = canMutate && visibleStatus === 'running';
  const canForceKill = me.data?.role === 'administrator' && ['starting', 'running', 'stopping'].includes(visibleStatus ?? '');
  const clientSnippet = pythonClientSnippet(endpoint ?? 'ws://HOST:PORT', '<API_KEY>');

  return (
    <div className="page deployment-detail-page">
      <PageIntro
        title={deployment.data?.name ?? t('deployment.details')}
        subtitle={deployment.data?.combination_id ?? deploymentId}
        actions={(
          <Space wrap>
            <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/deployments')}>{t('deployment.title')}</Button>
            {canCancel ? <Button danger icon={<CloseCircleOutlined />} loading={cancel.isPending} onClick={() => cancel.mutate()}>{t('deployment.cancelQueue')}</Button> : null}
            {canStop ? <Button danger icon={<StopOutlined />} loading={stop.isPending} onClick={() => stop.mutate()}>{t('deployment.stop')}</Button> : null}
            {canRestart ? <Button type="primary" icon={<ReloadOutlined />} loading={restart.isPending} onClick={() => restart.mutate()}>{t('deployment.restart')}</Button> : null}
            {canRotate ? (
              <Button
                icon={<KeyOutlined />}
                loading={rotate.isPending}
                onClick={() => Modal.confirm({
                  title: t('deployment.rotateKey'),
                  content: t('deployment.rotateKeyConfirm'),
                  onOk: () => rotate.mutateAsync(),
                })}
              >
                {t('deployment.rotateKey')}
              </Button>
            ) : null}
            {canForceKill ? (
              <Button
                type="primary"
                danger
                icon={<StopOutlined />}
                loading={forceKill.isPending}
                onClick={() => Modal.confirm({
                  title: t('deployment.forceKill'),
                  content: t('deployment.forceKillConfirm'),
                  okButtonProps: { danger: true },
                  onOk: () => forceKill.mutateAsync(),
                })}
              >
                {t('deployment.forceKill')}
              </Button>
            ) : null}
          </Space>
        )}
      />
      <AsyncState loading={deployment.isLoading || me.isLoading} error={deployment.error ?? me.error} onRetry={() => { void deployment.refetch(); void me.refetch(); }}>
        <div className="job-summary">
          <Card><Statistic title={t('common.status')} valueRender={() => <StatusTag status={visibleStatus} />} prefix={visibleStatus === 'running' ? <ReloadOutlined spin /> : <CheckCircleOutlined />} /></Card>
          <Card><Statistic title={t('deployment.uptime')} value={elapsed(deployment.data)} prefix={<ClockCircleOutlined />} /></Card>
          <Card><Statistic title={t('builder.gpuIds')} value={deployment.data?.assigned_gpu_ids.join(', ') || '—'} prefix={<LinkOutlined />} /></Card>
          <Card><Statistic title={t('jobs.queue')} value={deployment.data?.queue_position ? `#${deployment.data.queue_position}` : '—'} /></Card>
        </div>
        {deployment.data?.error_summary ? <Alert className="detail-alert" type="error" showIcon message={deployment.data.error_summary} /> : null}
        <Card className="job-tabs-card">
          <Tabs
            defaultActiveKey="overview"
            items={[
              {
                key: 'overview', label: t('jobs.overview'),
                children: (
                  <div className="overview-grid">
                    <Descriptions bordered column={{ xs: 1, md: 2 }}>
                      <Descriptions.Item label={t('common.owner')}>{deployment.data?.owner_name ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label="PID">{deployment.data?.pid ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label="Backbone">{deployment.data?.backbone_id ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label="Action Head">{deployment.data?.action_head_id ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label="Adapter">{deployment.data?.adapter_id ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('deployment.modelCombination')}>{deployment.data?.combination_id ?? '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('deployment.checkpoint')} span={2}><Typography.Text code>{deployment.data?.checkpoint_path ?? deployment.data?.checkpoint_id ?? '—'}</Typography.Text></Descriptions.Item>
                      <Descriptions.Item label={t('deployment.endpoint')} span={2}>
                        {endpoint ? <Space><Typography.Text code>{endpoint}</Typography.Text><Button size="small" type="text" icon={<CopyOutlined />} onClick={() => copy(endpoint)} /></Space> : '—'}
                      </Descriptions.Item>
                      <Descriptions.Item label={t('deployment.scope')}>{deployment.data?.endpoint.scope === 'lan' ? t('deployment.lanScope') : t('deployment.localScope')}</Descriptions.Item>
                      <Descriptions.Item label={t('deployment.idleTimeout')}>{deployment.data?.idle_timeout_seconds === -1 ? t('deployment.neverStop') : t('deployment.seconds', { value: deployment.data?.idle_timeout_seconds ?? 0 })}</Descriptions.Item>
                      <Descriptions.Item label={t('common.createdAt')}>{deployment.data?.created_at ? dayjs(deployment.data.created_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('jobs.started')}>{deployment.data?.started_at ? dayjs(deployment.data.started_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
                    </Descriptions>
                    <div>
                      <Space><Typography.Text strong>{t('deployment.clientExample')}</Typography.Text><Button type="text" size="small" icon={<CopyOutlined />} onClick={() => copy(clientSnippet)} /></Space>
                      <Alert className="deployment-client-hint" type="info" showIcon message={t('deployment.clientKeyPlaceholder')} />
                      <pre className="command-preview">{clientSnippet}</pre>
                    </div>
                    {Object.keys(deployment.data?.parameters ?? {}).length ? (
                      <div><Typography.Text strong>{t('deployment.adapterParameters')}</Typography.Text><pre className="config-preview">{JSON.stringify(deployment.data?.parameters, null, 2)}</pre></div>
                    ) : null}
                  </div>
                ),
              },
              {
                key: 'logs', label: t('jobs.logs'),
                children: (
                  <div>
                    <div className="terminal-toolbar">
                      <Tag icon={stream.connected || stream.ended ? <CheckCircleOutlined /> : <DisconnectOutlined />} color={stream.ended ? 'default' : stream.connected ? 'green' : 'orange'}>
                        {stream.ended ? t('jobs.streamEnded') : stream.connected ? t('jobs.streamConnected') : t('jobs.streamDisconnected')}
                      </Tag>
                      <Space><span>{t('jobs.follow')}</span><Switch size="small" checked={follow} onChange={setFollow} /></Space>
                    </div>
                    <div className="terminal" ref={terminalRef}>
                      {stream.logs.length ? stream.logs.map((line, index) => <div key={`${index}-${line.slice(0, 16)}`}><span className="line-no">{String(index + 1).padStart(4, '0')}</span>{line}</div>) : <span className="terminal-empty">{t('jobs.noLogs')}</span>}
                    </div>
                  </div>
                ),
              },
            ]}
          />
        </Card>
      </AsyncState>
      <DeploymentApiKeyModal
        open={Boolean(secret?.api_key)}
        deployment={secret?.deployment}
        apiKey={secret?.api_key}
        onClose={() => {
          setSecret(undefined);
          rotate.reset();
          restart.reset();
        }}
      />
    </div>
  );
}
