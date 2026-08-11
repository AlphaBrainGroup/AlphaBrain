import { ArrowLeftOutlined, CloudUploadOutlined, DownloadOutlined, FileZipOutlined, MergeCellsOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  Modal,
  Row,
  Select,
  Space,
  Switch,
  Tag,
  Typography,
  message,
} from 'antd';
import dayjs from 'dayjs';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router-dom';
import { api, healthyGpus, listFrom } from '../api/client';
import type { ModelPublication } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { UtilityProgress } from '../components/UtilityProgress';
import { useGpuRefreshInterval } from '../hooks/useGpuRefreshInterval';

interface PublishValues {
  repo_id: string;
  revision: string;
  private: boolean;
}

interface MergeValues {
  model: string;
  output_name?: string;
  gpu_id?: number;
}

export function CheckpointDetailPage() {
  const { checkpointId = '' } = useParams();
  const { t } = useTranslation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const gpuRefreshInterval = useGpuRefreshInterval();
  const [publishOpen, setPublishOpen] = useState(false);
  const [mergeOpen, setMergeOpen] = useState(false);
  const [publishForm] = Form.useForm<PublishValues>();
  const [mergeForm] = Form.useForm<MergeValues>();
  const checkpoint = useQuery({ queryKey: ['checkpoint', checkpointId], queryFn: () => api.checkpointDetail(checkpointId), enabled: Boolean(checkpointId) });
  const publications = useQuery({ queryKey: ['model-publications', checkpointId], queryFn: () => api.publications.list(checkpointId), enabled: Boolean(checkpointId), refetchInterval: 10_000 });
  const utilities = useQuery({ queryKey: ['utilities'], queryFn: api.utilities.list, refetchInterval: 2_000 });
  const gpus = useQuery({ queryKey: ['gpus'], queryFn: api.gpus, refetchInterval: gpuRefreshInterval });
  const visibleGpus = healthyGpus(gpus.data);
  const mergeModels = checkpoint.data?.tools.merge_lora.models ?? [];
  const suggestedName = useMemo(() => {
    const path = checkpoint.data?.tools.merge_lora.suggested_output_path;
    return path ? path.split('/').at(-1) : undefined;
  }, [checkpoint.data]);
  const packageRun = useMemo(() => listFrom(utilities.data).find(
    (run) => run.kind === 'checkpoint_package' && String(run.parameters?.checkpoint_id ?? '') === checkpointId,
  ), [checkpointId, utilities.data]);

  const publish = useMutation({
    mutationFn: (values: PublishValues) => api.publications.create({ checkpoint_id: checkpointId, ...values }),
    onSuccess: async () => {
      setPublishOpen(false);
      publishForm.resetFields();
      message.success(t('checkpointDetail.publishQueued'));
      await queryClient.invalidateQueries({ queryKey: ['model-publications', checkpointId] });
    },
    onError: (error) => message.error(error.message),
  });
  const merge = useMutation({
    mutationFn: (values: MergeValues) => api.mergeLoraCheckpoint(checkpointId, {
      model: values.model,
      output_name: values.output_name || undefined,
      resources: values.gpu_id == null
        ? { strategy: 'auto', gpu_count: 1, gpu_ids: [] }
        : { strategy: 'fixed', gpu_count: 1, gpu_ids: [values.gpu_id] },
    }),
    onSuccess: async () => {
      setMergeOpen(false);
      mergeForm.resetFields();
      message.success(t('checkpointDetail.mergeQueued'));
      await queryClient.invalidateQueries({ queryKey: ['utilities'] });
    },
    onError: (error) => message.error(error.message),
  });
  const packageCheckpoint = useMutation({
    mutationFn: () => api.packageCheckpoint(checkpointId),
    onSuccess: async () => {
      message.success(t('checkpoints.packageQueued'));
      await queryClient.invalidateQueries({ queryKey: ['utilities'] });
    },
    onError: (error) => message.error(error.message),
  });
  const cancelPackage = useMutation({
    mutationFn: () => api.utilities.cancel(packageRun!.id),
    onSuccess: async () => {
      message.success(t('checkpoints.packageCancelled'));
      await queryClient.invalidateQueries({ queryKey: ['utilities'] });
    },
    onError: (error) => message.error(error.message),
  });

  return (
    <div className="page">
      <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/checkpoints')}>{t('checkpointDetail.back')}</Button>
      <PageIntro title={checkpoint.data?.name || t('checkpointDetail.title')} subtitle={checkpoint.data?.experiment_name || t('checkpointDetail.subtitle')} />
      <AsyncState loading={checkpoint.isLoading} error={checkpoint.error} onRetry={() => void checkpoint.refetch()}>
        {checkpoint.data ? (
          <>
            <Card title={t('checkpointDetail.overview')}>
              <Descriptions column={{ xs: 1, md: 2 }}>
                <Descriptions.Item label={t('checkpoints.experiment')}>{checkpoint.data.experiment_name || '—'}</Descriptions.Item>
                <Descriptions.Item label={t('checkpoints.step')}>{checkpoint.data.step?.toLocaleString() ?? '—'}</Descriptions.Item>
                <Descriptions.Item label={t('checkpoints.complete')}><Tag color={checkpoint.data.complete ? 'green' : 'red'}>{checkpoint.data.complete ? t('checkpoints.integrityOk') : t('checkpoints.incomplete')}</Tag></Descriptions.Item>
                <Descriptions.Item label={t('checkpoints.resumable')}>{checkpoint.data.resumable ? t('common.yes') : t('common.no')}</Descriptions.Item>
                <Descriptions.Item label={t('checkpoints.path')} span={2}><Typography.Text code copyable>{checkpoint.data.path}</Typography.Text></Descriptions.Item>
              </Descriptions>
            </Card>
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col xs={24} lg={8}>
                <Card title={t('checkpointDetail.publishTitle')} extra={<CloudUploadOutlined />}>
                  <Typography.Paragraph>{t('checkpointDetail.publishDescription')}</Typography.Paragraph>
                  <Button type="primary" disabled={!checkpoint.data.tools.publish_huggingface.available} onClick={() => setPublishOpen(true)}>{t('checkpointDetail.publish')}</Button>
                </Card>
              </Col>
              <Col xs={24} lg={8}>
                <Card title={t('checkpointDetail.mergeTitle')} extra={<MergeCellsOutlined />}>
                  <Typography.Paragraph>{t('checkpointDetail.mergeDescription')}</Typography.Paragraph>
                  {!checkpoint.data.tools.merge_lora.available ? <Alert type="info" showIcon message={t('checkpointDetail.mergeUnavailable')} description={checkpoint.data.tools.merge_lora.reason} /> : null}
                  <Button style={{ marginTop: 12 }} disabled={!checkpoint.data.tools.merge_lora.available || checkpoint.data.tools.merge_lora.output_exists} onClick={() => { mergeForm.setFieldsValue({ model: mergeModels[0], output_name: suggestedName }); setMergeOpen(true); }}>{t('checkpointDetail.merge')}</Button>
                </Card>
              </Col>
              <Col xs={24} lg={8}>
                <Card title={t('checkpointDetail.packageTitle')} extra={<FileZipOutlined />}>
                  <Typography.Paragraph>{t('checkpointDetail.packageDescription')}</Typography.Paragraph>
                  {packageRun && ['queued', 'starting', 'running', 'stopping'].includes(packageRun.status) ? (
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <UtilityProgress run={packageRun} />
                      <Button danger loading={cancelPackage.isPending} onClick={() => cancelPackage.mutate()}>{t('common.cancel')}</Button>
                    </Space>
                  ) : (
                    <Space wrap>
                      {packageRun?.status === 'completed' ? <Button icon={<DownloadOutlined />} href={api.utilities.outputUrl(packageRun.id)}>{t('common.download')}</Button> : null}
                      <Button type="primary" icon={<FileZipOutlined />} disabled={!checkpoint.data.can_package} loading={packageCheckpoint.isPending} onClick={() => packageCheckpoint.mutate()}>
                        {packageRun?.status === 'completed' ? t('checkpoints.repackage') : t('checkpoints.package')}
                      </Button>
                    </Space>
                  )}
                </Card>
              </Col>
            </Row>
            <Card title={t('checkpointDetail.inspection')} style={{ marginTop: 16 }}>
              <pre className="config-preview">{JSON.stringify(checkpoint.data.inspection, null, 2)}</pre>
            </Card>
            <Card title={t('checkpointDetail.publicationHistory')} style={{ marginTop: 16 }}>
              <AsyncState loading={publications.isLoading} error={publications.error} empty={!listFrom(publications.data).length} onRetry={() => void publications.refetch()}>
                <Table<ModelPublication>
                  rowKey="id"
                  dataSource={listFrom(publications.data)}
                  pagination={false}
                  columns={[
                    { title: t('checkpointDetail.repository'), dataIndex: 'repo_id', render: (value: string, row) => row.result_url ? <Typography.Link href={row.result_url} target="_blank" rel="noreferrer">{value}</Typography.Link> : value },
                    { title: t('checkpointDetail.revision'), dataIndex: 'revision' },
                    { title: t('common.status'), dataIndex: 'status', render: (value: string) => <Tag color={value === 'completed' ? 'green' : value === 'failed' ? 'red' : 'blue'}>{value}</Tag> },
                    { title: t('checkpointDetail.visibility'), dataIndex: 'private', render: (value: boolean) => value ? t('checkpointDetail.private') : t('checkpointDetail.public') },
                    { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm') : '—' },
                  ]}
                />
              </AsyncState>
            </Card>
          </>
        ) : null}
      </AsyncState>
      <Modal title={t('checkpointDetail.publishTitle')} open={publishOpen} onCancel={() => setPublishOpen(false)} onOk={() => publishForm.submit()} confirmLoading={publish.isPending} destroyOnHidden>
        <Form<PublishValues> form={publishForm} layout="vertical" initialValues={{ revision: 'main', private: true }} onFinish={(values) => publish.mutate(values)}>
          <Form.Item name="repo_id" label={t('checkpointDetail.repository')} rules={[{ required: true }, { pattern: /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/, message: t('checkpointDetail.repositoryHint') }]}>
            <Input placeholder="organization/model-name" />
          </Form.Item>
          <Form.Item name="revision" label={t('checkpointDetail.revision')} rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="private" label={t('checkpointDetail.visibility')} valuePropName="checked"><Switch checkedChildren={t('checkpointDetail.private')} unCheckedChildren={t('checkpointDetail.public')} /></Form.Item>
          <Alert type="info" showIcon message={t('checkpointDetail.tokenHint')} />
        </Form>
      </Modal>
      <Modal title={t('checkpointDetail.mergeTitle')} open={mergeOpen} onCancel={() => setMergeOpen(false)} onOk={() => mergeForm.submit()} confirmLoading={merge.isPending} destroyOnHidden>
        <Form<MergeValues> form={mergeForm} layout="vertical" onFinish={(values) => merge.mutate(values)}>
          <Form.Item name="model" label={t('checkpointDetail.modelRecipe')} rules={[{ required: true }]}><Select options={mergeModels.map((value) => ({ value, label: value }))} /></Form.Item>
          <Form.Item name="output_name" label={t('checkpointDetail.outputName')} extra={t('checkpointDetail.outputHint')}><Input /></Form.Item>
          {visibleGpus.length > 1 ? (
            <Form.Item name="gpu_id" label={t('checkpointDetail.gpu')} extra={t('checkpointDetail.gpuHint')}>
              <Select allowClear placeholder={t('checkpointDetail.autoGpu')} options={visibleGpus.map((gpu) => ({ value: gpu.index, label: `GPU ${gpu.index} · ${gpu.name}`, disabled: !gpu.available }))} />
            </Form.Item>
          ) : (
            <Alert type="info" showIcon message={visibleGpus.length === 1 ? t('builder.singleGpuDetected') : t('builder.noGpuDetected')} />
          )}
        </Form>
      </Modal>
    </div>
  );
}
