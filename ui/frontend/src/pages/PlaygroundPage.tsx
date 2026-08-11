import { DeleteOutlined, InboxOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Form,
  Input,
  Row,
  Select,
  Space,
  Tag,
  Typography,
  Upload,
  message,
  type UploadFile,
  type UploadProps,
} from 'antd';
import dayjs from 'dayjs';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api, listFrom } from '../api/client';
import type { InferenceRun } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';

function statusColor(status: InferenceRun['status']): string {
  if (status === 'completed') return 'green';
  if (status === 'running') return 'blue';
  if (status === 'timed_out') return 'orange';
  return 'red';
}

export function PlaygroundPage() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [deploymentId, setDeploymentId] = useState<string>();
  const [instructions, setInstructions] = useState('');
  const [states, setStates] = useState('');
  const [saveInputs, setSaveInputs] = useState(false);
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [result, setResult] = useState<InferenceRun>();

  const deployments = useQuery({ queryKey: ['deployments'], queryFn: api.deployments.list, refetchInterval: 10_000 });
  const running = useMemo(
    () => listFrom(deployments.data).filter((item) => item.status === 'running'),
    [deployments.data],
  );
  useEffect(() => {
    if (!deploymentId && running.length) setDeploymentId(running[0].id);
    if (deploymentId && !running.some((item) => item.id === deploymentId)) setDeploymentId(running[0]?.id);
  }, [deploymentId, running]);
  const selected = running.find((item) => item.id === deploymentId);
  const history = useQuery({
    queryKey: ['inference-runs', deploymentId],
    queryFn: () => api.inferenceRuns.list(deploymentId),
    enabled: Boolean(deploymentId),
    refetchInterval: 10_000,
  });

  const instructionBatch = useMemo(
    () => instructions.split('\n').map((item) => item.trim()).filter(Boolean),
    [instructions],
  );
  const stateValid = useMemo(() => {
    if (!states.trim()) return true;
    try {
      return Array.isArray(JSON.parse(states));
    } catch {
      return false;
    }
  }, [states]);
  const imageFiles = files.flatMap((file) => file.originFileObj instanceof File ? [file.originFileObj] : []);
  const canRun = Boolean(
    selected
    && selected.managed_inference_available !== false
    && instructionBatch.length >= 1
    && instructionBatch.length <= 8
    && stateValid
    && imageFiles.length <= 4,
  );

  const infer = useMutation({
    mutationFn: () => api.deployments.infer(deploymentId!, {
      instructions: instructionBatch,
      states,
      images: imageFiles,
      save_inputs: saveInputs,
    }),
    onSuccess: async (value) => {
      setResult(value);
      message.success(t('playground.completed'));
      await queryClient.invalidateQueries({ queryKey: ['inference-runs', deploymentId] });
      await queryClient.invalidateQueries({ queryKey: ['deployments'] });
    },
    onError: (error) => message.error(error.message),
  });

  const uploadProps: UploadProps = {
    accept: 'image/jpeg,image/png,image/webp',
    multiple: true,
    maxCount: 4,
    fileList: files,
    beforeUpload: (file) => {
      if (file.size > 10 * 1024 * 1024) {
        message.error(t('playground.imageTooLarge'));
        return Upload.LIST_IGNORE;
      }
      return false;
    },
    onChange: ({ fileList }) => setFiles(fileList.slice(0, 4)),
    itemRender: (originNode, _file, _fileList, actions) => (
      <Space>
        {originNode}
        <Button type="text" danger size="small" icon={<DeleteOutlined />} onClick={actions.remove} aria-label={t('common.delete')} />
      </Space>
    ),
  };

  return (
    <div className="page">
      <PageIntro title={t('playground.title')} subtitle={t('playground.subtitle')} />
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={13}>
          <Card title={t('playground.request')}>
            <Form layout="vertical" onFinish={() => infer.mutate()}>
              <Form.Item label={t('playground.deployment')} required>
                <Select
                  value={deploymentId}
                  onChange={(value) => { setDeploymentId(value); setResult(undefined); }}
                  loading={deployments.isLoading}
                  placeholder={t('playground.selectDeployment')}
                  options={running.map((item) => ({
                    value: item.id,
                    label: `${item.name} · ${item.combination_id}`,
                  }))}
                  notFoundContent={t('playground.noRunningDeployment')}
                />
              </Form.Item>
              {selected?.managed_inference_available === false ? (
                <Alert type="warning" showIcon message={t('playground.controllerUnavailable')} description={t('playground.controllerUnavailableHint')} />
              ) : null}
              <Form.Item
                label={t('playground.instructions')}
                required
                validateStatus={instructionBatch.length > 8 ? 'error' : undefined}
                help={instructionBatch.length > 8 ? t('playground.batchTooLarge') : t('playground.instructionsHint')}
              >
                <Input.TextArea
                  rows={5}
                  value={instructions}
                  onChange={(event) => setInstructions(event.target.value)}
                  placeholder={t('playground.instructionsPlaceholder')}
                  maxLength={8 * 4096}
                  showCount
                />
              </Form.Item>
              <Form.Item label={t('playground.images')} extra={t('playground.imagesHint')}>
                <Upload.Dragger {...uploadProps}>
                  <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                  <p className="ant-upload-text">{t('playground.uploadImages')}</p>
                  <p className="ant-upload-hint">{t('playground.imageLimits')}</p>
                </Upload.Dragger>
              </Form.Item>
              <Form.Item
                label={t('playground.states')}
                validateStatus={stateValid ? undefined : 'error'}
                help={stateValid ? t('playground.statesHint') : t('playground.invalidStates')}
              >
                <Input.TextArea
                  rows={3}
                  value={states}
                  onChange={(event) => setStates(event.target.value)}
                  placeholder="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
                />
              </Form.Item>
              <Form.Item>
                <Checkbox checked={saveInputs} onChange={(event) => setSaveInputs(event.target.checked)}>
                  {t('playground.saveInputs')}
                </Checkbox>
                <Typography.Paragraph type="secondary" className="form-hint">
                  {t('playground.saveInputsHint')}
                </Typography.Paragraph>
              </Form.Item>
              <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} loading={infer.isPending} disabled={!canRun}>
                {t('playground.run')}
              </Button>
            </Form>
          </Card>
        </Col>
        <Col xs={24} xl={11}>
          <Card title={t('playground.output')}>
            {result ? (
              <>
                <Space wrap>
                  <Tag color={statusColor(result.status)}>{t(`status.${result.status}`, { defaultValue: result.status })}</Tag>
                  <Typography.Text>{t('playground.latency')}: {result.latency_ms?.toFixed(1) ?? '—'} ms</Typography.Text>
                  <Typography.Text>{t('playground.batch')}: {result.batch_size}</Typography.Text>
                </Space>
                <pre className="config-preview playground-output">{JSON.stringify(result.output, null, 2)}</pre>
              </>
            ) : (
              <div className="empty-compact"><Typography.Text type="secondary">{t('playground.waitingOutput')}</Typography.Text></div>
            )}
          </Card>
        </Col>
      </Row>
      <Card title={t('playground.history')} style={{ marginTop: 16 }}>
        <AsyncState loading={history.isLoading} error={history.error} empty={!listFrom(history.data).length} onRetry={() => void history.refetch()}>
          <Table<InferenceRun>
            rowKey="id"
            dataSource={listFrom(history.data)}
            pagination={{ pageSize: 10 }}
            columns={[
              { title: t('common.status'), dataIndex: 'status', render: (value: InferenceRun['status']) => <Tag color={statusColor(value)}>{t(`status.${value}`, { defaultValue: value })}</Tag> },
              { title: t('playground.batch'), dataIndex: 'batch_size' },
              { title: t('playground.images'), dataIndex: 'image_count' },
              { title: t('playground.latency'), dataIndex: 'latency_ms', render: (value?: number) => value == null ? '—' : `${value.toFixed(1)} ms` },
              { title: t('playground.inputsSaved'), dataIndex: 'inputs_saved', render: (value: boolean) => value ? t('common.yes') : t('common.no') },
              { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm:ss') : '—' },
            ]}
          />
        </AsyncState>
      </Card>
    </div>
  );
}
