import { ArrowLeftOutlined, ApiOutlined, CopyOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Button, Card, Descriptions, Space, Steps, message } from 'antd';
import dayjs from 'dayjs';
import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router-dom';
import YAML from 'yaml';
import { api } from '../api/client';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { StatusTag } from '../components/StatusTag';

export function ExperimentDetailPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { experimentId } = useParams();
  const experiment = useQuery({ queryKey: ['experiment', experimentId], queryFn: () => api.experiments.get(experimentId!), enabled: Boolean(experimentId) });
  return (
    <div className="page">
      <PageIntro
        title={experiment.data?.name ?? t('checkpoints.experiment')}
        subtitle={experiment.data?.description ?? experimentId}
        actions={<Space><Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/')}>{t('nav.dashboard')}</Button>{experiment.data?.job_id ? <Button type="primary" icon={<ApiOutlined />} onClick={() => navigate(`/jobs/${experiment.data?.job_id}`)}>{t('jobs.details')}</Button> : null}</Space>}
      />
      <AsyncState loading={experiment.isLoading} error={experiment.error} onRetry={() => void experiment.refetch()}>
        <div className="experiment-detail-grid">
          <Card title={t('jobs.overview')}>
            <Descriptions bordered column={1}>
              <Descriptions.Item label={t('common.status')}><StatusTag status={experiment.data?.status} /></Descriptions.Item>
              <Descriptions.Item label={t('common.owner')}>{experiment.data?.owner_name ?? '—'}</Descriptions.Item>
              <Descriptions.Item label={t('builder.backbone')}>{experiment.data?.architecture ?? '—'}</Descriptions.Item>
              <Descriptions.Item label={t('builder.method')}>{experiment.data?.method ?? '—'}</Descriptions.Item>
              <Descriptions.Item label={t('builder.dataset')}>{experiment.data?.dataset ?? '—'}</Descriptions.Item>
              <Descriptions.Item label={t('common.createdAt')}>{experiment.data?.created_at ? dayjs(experiment.data.created_at).format('YYYY-MM-DD HH:mm:ss') : '—'}</Descriptions.Item>
            </Descriptions>
            {experiment.data?.stages?.length ? <Steps direction="vertical" className="stage-steps" items={experiment.data.stages.map((stage) => ({ title: stage.name, description: <StatusTag status={stage.status} />, status: stage.status === 'completed' ? 'finish' : stage.status === 'running' ? 'process' : stage.status === 'failed' ? 'error' : 'wait', onClick: () => stage.job_id && navigate(`/jobs/${stage.job_id}`) }))} /> : null}
          </Card>
          <Card title={<Space>{t('builder.resolvedConfig')}<Button type="text" size="small" icon={<CopyOutlined />} onClick={() => { void navigator.clipboard.writeText(YAML.stringify(experiment.data?.config ?? {})); message.success(t('common.copied')); }} /></Space>}>
            <pre className="config-preview large">{YAML.stringify(experiment.data?.config ?? {})}</pre>
          </Card>
        </div>
      </AsyncState>
    </div>
  );
}
