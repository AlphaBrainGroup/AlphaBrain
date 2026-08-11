import {
  CloudServerOutlined,
  CopyOutlined,
  ExperimentOutlined,
  PlusOutlined,
  SafetyCertificateOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  Space,
  Steps,
  Switch,
  Tag,
  Typography,
  message,
} from 'antd';
import dayjs from 'dayjs';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { api, healthyGpus } from '../api/client';
import type {
  DeploymentCreateRequest,
  DeploymentMutationResult,
  Checkpoint,
  ModelDeployment,
  ParameterSchema,
  PreflightResult,
} from '../api/types';
import { usePreferences } from '../app-context';
import { useGpuRefreshInterval } from '../hooks/useGpuRefreshInterval';
import { AsyncState } from '../components/AsyncState';
import { DeploymentApiKeyModal } from '../components/DeploymentApiKeyModal';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { ServerDirectoryPicker } from '../components/ServerDirectoryPicker';
import { StatusTag } from '../components/StatusTag';
import {
  deploymentEndpoint,
  mergeParameterSchemas,
  normalizeDeploymentResources,
  parameterDefaults,
} from './deployment-form';

interface DeploymentFormValues extends DeploymentCreateRequest {
  checkpoint_kind: 'indexed' | 'local';
  checkpoint_id?: string;
  checkpoint_path?: string;
}

function localized(language: string, english?: string, chinese?: string, fallback = ''): string {
  return language === 'zh-CN' ? chinese || english || fallback : english || chinese || fallback;
}

function isSet(value: unknown): boolean {
  return value !== undefined && value !== null && value !== '';
}

function DynamicParameter({ schema }: { schema: ParameterSchema }) {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const label = localized(language, schema.label, schema.label_zh, schema.key);
  const description = localized(language, schema.description, schema.description_zh);
  const rules = schema.required && schema.type !== 'boolean'
    ? [{ required: true, message: t('deployment.parameterRequired', { name: label }) }]
    : undefined;
  let control;
  if (schema.type === 'boolean') control = <Switch />;
  else if (schema.type === 'number' || schema.type === 'integer') {
    control = <InputNumber className="full-width" min={schema.min} max={schema.max} precision={schema.type === 'integer' ? 0 : undefined} />;
  } else if (schema.type === 'select') {
    control = (
      <Select
        options={(schema.options ?? []).map((option) => ({
          value: option.value,
          label: localized(language, option.label, option.label_zh, String(option.value)),
        }))}
      />
    );
  } else if (schema.type === 'textarea') control = <Input.TextArea autoSize={{ minRows: 3, maxRows: 8 }} />;
  else if (schema.type === 'password') control = <Input.Password autoComplete="off" />;
  else if (schema.type === 'path') control = <ServerDirectoryPicker placeholder="/path/to/resource" />;
  else control = <Input />;
  return (
    <Form.Item
      name={['parameters', schema.key]}
      label={label}
      extra={description || undefined}
      rules={rules}
      valuePropName={schema.type === 'boolean' ? 'checked' : 'value'}
    >
      {control}
    </Form.Item>
  );
}

function CreateDeploymentModal({ open, onClose, onCreated }: {
  open: boolean;
  onClose: () => void;
  onCreated: (result: DeploymentMutationResult) => void;
}) {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const gpuRefreshInterval = useGpuRefreshInterval();
  const [form] = Form.useForm<DeploymentFormValues>();
  const [step, setStep] = useState(0);
  const [preflightResult, setPreflightResult] = useState<PreflightResult>();
  const [searchParams] = useSearchParams();
  const capabilities = useQuery({ queryKey: ['deployment-capabilities'], queryFn: api.deployments.capabilities, enabled: open });
  const checkpoints = useQuery({ queryKey: ['checkpoints'], queryFn: api.checkpoints, enabled: open });
  const gpus = useQuery({ queryKey: ['gpus'], queryFn: api.gpus, enabled: open, refetchInterval: open ? gpuRefreshInterval : false });
  const values = Form.useWatch([], form) as DeploymentFormValues | undefined;
  const visibleGpus = healthyGpus(gpus.data);
  const visibleGpuCount = visibleGpus.length;
  const singleGpu = visibleGpuCount === 1;
  const selectedCombination = capabilities.data?.combinations.find((item) => item.id === values?.combination_id);
  const selectedAdapter = capabilities.data?.adapters.find((item) => item.id === selectedCombination?.adapter_id);
  const parameters = useMemo(
    () => mergeParameterSchemas(selectedAdapter?.parameters, selectedCombination?.parameters),
    [selectedAdapter?.parameters, selectedCombination?.parameters],
  );

  useEffect(() => {
    if (!open) return;
    const checkpointId = searchParams.get('checkpoint_id') ?? undefined;
    const checkpointPath = searchParams.get('checkpoint') ?? undefined;
    const combinationId = searchParams.get('combination_id') ?? undefined;
    const requestedName = searchParams.get('name') ?? '';
    form.resetFields();
    form.setFieldsValue({
      name: requestedName,
      checkpoint_kind: checkpointId ? 'indexed' : checkpointPath ? 'local' : 'indexed',
      checkpoint_id: checkpointId,
      checkpoint_path: checkpointPath,
      combination_id: combinationId,
      resources: { strategy: 'auto', gpu_count: 1, gpu_ids: [] },
      endpoint: {
        scope: 'local',
        advertised_host: '127.0.0.1',
        idle_timeout_seconds: 1800,
      },
      parameters: {},
      acknowledge_experimental: false,
    });
    setStep(0);
    setPreflightResult(undefined);
  }, [form, open, searchParams]);

  useEffect(() => {
    if (!selectedCombination) return;
    form.setFieldValue('parameters', parameterDefaults(parameters));
    if (selectedCombination.recommended_gpu_count && !singleGpu) {
      form.setFieldValue(['resources', 'gpu_count'], selectedCombination.recommended_gpu_count);
    }
    setPreflightResult(undefined);
  }, [form, parameters, selectedCombination, singleGpu]);

  useEffect(() => {
    if (!open || !singleGpu) return;
    form.setFieldValue('resources', normalizeDeploymentResources(form.getFieldValue('resources'), visibleGpuCount));
    setPreflightResult(undefined);
  }, [form, open, singleGpu, visibleGpuCount]);

  const makePayload = async (): Promise<DeploymentCreateRequest> => {
    await form.validateFields();
    const data = form.getFieldsValue(true);
    const checkpoint_source = data.checkpoint_kind === 'indexed'
      ? { kind: 'indexed' as const, checkpoint_id: data.checkpoint_id! }
      : { kind: 'local' as const, path: data.checkpoint_path! };
    return {
      name: data.name.trim(),
      checkpoint_source,
      combination_id: data.combination_id,
      resources: normalizeDeploymentResources(data.resources, visibleGpuCount),
      endpoint: {
        scope: data.endpoint.scope,
        ...(data.endpoint.advertised_host?.trim() ? { advertised_host: data.endpoint.advertised_host.trim() } : {}),
        ...(data.endpoint.port ? { port: data.endpoint.port } : {}),
        idle_timeout_seconds: data.endpoint.idle_timeout_seconds,
      },
      parameters: data.parameters ?? {},
      acknowledge_experimental: Boolean(data.acknowledge_experimental),
    };
  };

  const preflight = useMutation({
    mutationFn: async () => api.deployments.preflight(await makePayload()),
    onSuccess: setPreflightResult,
    onError: (error) => message.error(error.message),
  });
  const create = useMutation({
    mutationFn: async () => api.deployments.create(await makePayload()),
    onSuccess: (result) => {
      message.success(t('deployment.created'));
      onCreated(result);
    },
    onError: (error) => message.error(error.message),
  });

  const sourceReady = Boolean(
    values?.name?.trim()
    && (values.checkpoint_kind === 'indexed' ? values.checkpoint_id : values?.checkpoint_path?.trim()),
  );
  const parametersReady = Boolean(
    selectedCombination
    && parameters.every((parameter) => !parameter.required || parameter.type === 'boolean' || isSet(values?.parameters?.[parameter.key]))
    && (selectedCombination.status !== 'experimental' || values?.acknowledge_experimental),
  );
  const requestedGpuCount = values?.resources?.gpu_count ?? 0;
  const idleTimeout = values?.endpoint?.idle_timeout_seconds;
  const resourceReady = Boolean(
    (singleGpu || (requestedGpuCount >= 1
      && (values?.resources?.strategy !== 'fixed'
        || (values?.resources?.gpu_ids?.length ?? 0) === requestedGpuCount)))
    && values?.endpoint?.scope
    && idleTimeout != null
    && idleTimeout >= -1,
  );
  const nextDisabled = step === 0 ? !sourceReady : step === 1 ? !parametersReady : step === 2 ? !resourceReady : true;

  const next = async () => {
    const fields: Parameters<typeof form.validateFields>[0] = step === 0
      ? [['name'], ['checkpoint_id'], ['checkpoint_path']]
      : step === 1 ? [['combination_id'], ...parameters.map((parameter) => ['parameters', parameter.key])]
        : [['resources'], ['endpoint']];
    try {
      await form.validateFields(fields);
      setStep((current) => Math.min(3, current + 1));
    } catch {
      // Ant Design renders field errors next to the relevant control.
    }
  };

  const checkpointKind = values?.checkpoint_kind ?? 'indexed';
  const normalizedResources = normalizeDeploymentResources(values?.resources, visibleGpuCount);
  const resourceStrategy = normalizedResources.strategy;
  const endpointScope = values?.endpoint?.scope ?? 'local';

  return (
    <Modal
      className="deployment-create-modal"
      open={open}
      width={920}
      title={t('deployment.createTitle')}
      onCancel={onClose}
      footer={(
        <div className="builder-footer">
          <Button disabled={step === 0} onClick={() => setStep((current) => Math.max(0, current - 1))}>{t('common.previous')}</Button>
          <Space>
            <Button onClick={onClose}>{t('common.cancel')}</Button>
            {step < 3 ? <Button type="primary" disabled={nextDisabled} onClick={() => void next()}>{t('common.next')}</Button> : null}
            {step === 3 ? <Button icon={<SafetyCertificateOutlined />} loading={preflight.isPending} onClick={() => preflight.mutate()}>{t('deployment.runPreflight')}</Button> : null}
            {step === 3 ? <Button type="primary" icon={<CloudServerOutlined />} disabled={!preflightResult?.ok} loading={create.isPending} onClick={() => create.mutate()}>{t('deployment.submit')}</Button> : null}
          </Space>
        </div>
      )}
    >
      <Steps
        current={step}
        onChange={(nextStep) => { if (nextStep < step) setStep(nextStep); }}
        items={[
          { title: t('deployment.sourceStep') },
          { title: t('deployment.modelStep') },
          { title: t('deployment.runtimeStep') },
          { title: t('deployment.reviewStep') },
        ]}
      />
      <AsyncState
        loading={capabilities.isLoading || checkpoints.isLoading || gpus.isLoading}
        error={capabilities.error ?? checkpoints.error ?? gpus.error}
        onRetry={() => { void capabilities.refetch(); void checkpoints.refetch(); void gpus.refetch(); }}
      >
        <Form<DeploymentFormValues>
          className="deployment-form"
          form={form}
          layout="vertical"
          onValuesChange={() => setPreflightResult(undefined)}
        >
          {step === 0 ? (
            <div className="deployment-step">
              <Form.Item name="name" label={t('deployment.name')} rules={[{ required: true, whitespace: true }]}>
                <Input placeholder={t('deployment.namePlaceholder')} />
              </Form.Item>
              <Form.Item name="checkpoint_kind" label={t('deployment.checkpointSource')}>
                <Radio.Group optionType="button" buttonStyle="solid" options={[
                  { label: t('deployment.indexedCheckpoint'), value: 'indexed' },
                  { label: t('deployment.localCheckpoint'), value: 'local' },
                ]} />
              </Form.Item>
              {checkpointKind === 'indexed' ? (
                <Form.Item name="checkpoint_id" label={t('deployment.checkpoint')} rules={[{ required: true }]}>
                  <Select
                    showSearch
                    optionFilterProp="searchText"
                    placeholder={t('deployment.selectCheckpoint')}
                    options={(checkpoints.data ?? []).filter((checkpoint) => checkpoint.complete !== false).map((checkpoint) => ({
                      value: checkpoint.id,
                      label: checkpoint.experiment_name ? `${checkpoint.experiment_name} · ${checkpoint.path}` : checkpoint.path,
                      searchText: `${checkpoint.experiment_name ?? ''} ${checkpoint.path}`,
                    }))}
                  />
                </Form.Item>
              ) : (
                <Form.Item name="checkpoint_path" label={t('deployment.checkpointPath')} rules={[{ required: true, whitespace: true }]}>
                  <ServerDirectoryPicker placeholder="/path/to/checkpoint" />
                </Form.Item>
              )}
              <Alert type="info" showIcon message={t('deployment.checkpointReadOnly')} />
            </div>
          ) : null}
          {step === 1 ? (
            <div className="deployment-step">
              <Form.Item name="combination_id" label={t('deployment.modelCombination')} rules={[{ required: true }]}>
                <Select
                  showSearch
                  optionFilterProp="searchText"
                  placeholder={t('deployment.selectCombination')}
                  options={(capabilities.data?.combinations ?? []).map((combination) => ({
                    value: combination.id,
                    label: localized(language, combination.name, combination.name_zh, combination.id),
                    searchText: `${combination.id} ${combination.name} ${combination.name_zh ?? ''} ${combination.backbone_id} ${combination.action_head_id}`,
                  }))}
                />
              </Form.Item>
              {selectedCombination ? (
                <Card size="small" className="deployment-combination-card">
                  <Space wrap>
                    <Tag color="blue">Backbone · {selectedCombination.backbone_id}</Tag>
                    <Tag color="purple">Action Head · {selectedCombination.action_head_id}</Tag>
                    <Tag>Adapter · {selectedCombination.adapter_id}</Tag>
                    <Tag color={selectedCombination.status === 'experimental' ? 'orange' : 'green'}>{t(`capability.${selectedCombination.status}`)}</Tag>
                  </Space>
                  {localized(language, selectedCombination.description, selectedCombination.description_zh) ? (
                    <Typography.Paragraph type="secondary">
                      {localized(language, selectedCombination.description, selectedCombination.description_zh)}
                    </Typography.Paragraph>
                  ) : null}
                </Card>
              ) : null}
              {selectedCombination?.status === 'experimental' ? (
                <>
                  <Alert type="warning" showIcon message={t('builder.experimental')} description={t('builder.experimentalRisk')} />
                  <Form.Item name="acknowledge_experimental" valuePropName="checked" rules={[{
                    validator: (_, checked) => checked ? Promise.resolve() : Promise.reject(new Error(t('deployment.experimentalConfirmationRequired'))),
                  }]}>
                    <Checkbox>{t('deployment.experimentalConfirmation')}</Checkbox>
                  </Form.Item>
                </>
              ) : null}
              {parameters.length ? <Typography.Title level={5}>{t('deployment.adapterParameters')}</Typography.Title> : null}
              <div className="form-grid-2">{parameters.map((parameter) => <DynamicParameter key={parameter.key} schema={parameter} />)}</div>
            </div>
          ) : null}
          {step === 2 ? (
            <div className="deployment-step">
              <Typography.Title level={5}>{t('deployment.gpuResources')}</Typography.Title>
              {singleGpu ? (
                <Alert
                  type="info"
                  showIcon
                  message={t('builder.singleGpuDetected')}
                  description={t('builder.singleGpuDescription', {
                    index: visibleGpus[0]?.index ?? 0,
                    name: visibleGpus[0]?.name ?? 'GPU',
                  })}
                />
              ) : (
                <>
                  <div className="form-grid-2">
                    <Form.Item name={['resources', 'strategy']} label={t('builder.resourceStrategy')}>
                      <Radio.Group options={[{ label: t('builder.autoGpu'), value: 'auto' }, { label: t('builder.fixedGpu'), value: 'fixed' }]} />
                    </Form.Item>
                    <Form.Item name={['resources', 'gpu_count']} label={t('builder.gpuCount')} rules={[{ required: true }]}>
                      <InputNumber className="full-width" min={1} max={Math.max(1, visibleGpus.length || 8)} precision={0} />
                    </Form.Item>
                  </div>
                  {resourceStrategy === 'fixed' ? (
                    <Form.Item
                      name={['resources', 'gpu_ids']}
                      label={t('builder.gpuIds')}
                      rules={[{
                        validator: (_, ids?: number[]) => ids?.length === values?.resources?.gpu_count
                          ? Promise.resolve()
                          : Promise.reject(new Error(t('deployment.gpuSelectionCount'))),
                      }]}
                    >
                      <Select mode="multiple" options={visibleGpus.map((gpu) => ({ value: gpu.index, label: `GPU ${gpu.index} · ${gpu.name}` }))} />
                    </Form.Item>
                  ) : null}
                </>
              )}
              <Alert type="info" showIcon message={t('deployment.fifoHint')} />
              <Typography.Title level={5}>{t('deployment.endpointSettings')}</Typography.Title>
              <div className="form-grid-2">
                <Form.Item name={['endpoint', 'scope']} label={t('deployment.endpointScope')}>
                  <Radio.Group
                    options={[{ label: t('deployment.localScope'), value: 'local' }, { label: t('deployment.lanScope'), value: 'lan' }]}
                    onChange={(event) => form.setFieldValue(['endpoint', 'advertised_host'], event.target.value === 'local' ? '127.0.0.1' : window.location.hostname)}
                  />
                </Form.Item>
                <Form.Item name={['endpoint', 'advertised_host']} label={t('deployment.advertisedHost')} extra={endpointScope === 'lan' ? t('deployment.advertisedHostHint') : undefined}>
                  <Input placeholder={endpointScope === 'lan' ? window.location.hostname : '127.0.0.1'} />
                </Form.Item>
                <Form.Item name={['endpoint', 'port']} label={t('deployment.port')} extra={t('deployment.portHint')} rules={[{ type: 'number', min: 1024, max: 65535 }]}>
                  <InputNumber className="full-width" min={1024} max={65535} precision={0} placeholder={t('deployment.autoPort')} />
                </Form.Item>
                <Form.Item name={['endpoint', 'idle_timeout_seconds']} label={t('deployment.idleTimeout')} extra={t('deployment.idleTimeoutHint')} rules={[{ required: true, type: 'number', min: -1 }]}>
                  <InputNumber className="full-width" min={-1} precision={0} />
                </Form.Item>
              </div>
            </div>
          ) : null}
          {step === 3 ? (
            <div className="deployment-step">
              <Descriptions bordered column={1} size="small">
                <Descriptions.Item label={t('deployment.name')}>{values?.name}</Descriptions.Item>
                <Descriptions.Item label={t('deployment.checkpoint')}>{checkpointKind === 'indexed' ? values?.checkpoint_id : values?.checkpoint_path}</Descriptions.Item>
                <Descriptions.Item label={t('deployment.modelCombination')}>{selectedCombination ? localized(language, selectedCombination.name, selectedCombination.name_zh, selectedCombination.id) : values?.combination_id}</Descriptions.Item>
                <Descriptions.Item label={t('deployment.gpuResources')}>{resourceStrategy === 'fixed' ? `GPU ${(normalizedResources.gpu_ids ?? []).join(', ')}` : `${t('builder.autoGpu')} · ${normalizedResources.gpu_count}`}</Descriptions.Item>
                <Descriptions.Item label={t('deployment.endpointSettings')}>{endpointScope === 'local' ? t('deployment.localScope') : t('deployment.lanScope')} · {values?.endpoint?.advertised_host || t('common.unknown')}:{values?.endpoint?.port || t('deployment.autoPort')}</Descriptions.Item>
              </Descriptions>
              {preflightResult ? (
                <div className="deployment-preflight">
                  <Alert type={preflightResult.ok ? 'success' : 'error'} showIcon message={preflightResult.ok ? t('deployment.preflightPassed') : t('deployment.preflightFailed')} />
                  {preflightResult.items.map((item, index) => (
                    <Alert
                      key={`${item.id ?? item.title}-${index}`}
                      type={item.level === 'error' ? 'error' : item.level === 'warning' ? 'warning' : item.level === 'success' ? 'success' : 'info'}
                      showIcon
                      message={item.message_i18n?.[language] ?? item.message ?? t(`preflight.${item.id ?? 'unknown'}`)}
                    />
                  ))}
                </div>
              ) : <Alert className="deployment-preflight" type="info" showIcon message={t('deployment.preflightRequired')} />}
            </div>
          ) : null}
        </Form>
      </AsyncState>
    </Modal>
  );
}

export function DeploymentsPage() {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [search, setSearch] = useState('');
  const [secret, setSecret] = useState<DeploymentMutationResult>();
  const deployments = useQuery({ queryKey: ['deployments'], queryFn: api.deployments.list, refetchInterval: 5_000 });
  const checkpoints = useQuery({ queryKey: ['checkpoints'], queryFn: api.checkpoints });
  const localPresets = useMemo(() => (checkpoints.data ?? []).filter((item) => item.builtin && item.deployable), [checkpoints.data]);
  const createOpen = searchParams.get('create') === '1';
  const rows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return (deployments.data ?? []).filter((deployment) => !term || [
      deployment.name,
      deployment.owner_name,
      deployment.combination_id,
      deployment.checkpoint_path,
      deploymentEndpoint(deployment),
    ].join(' ').toLowerCase().includes(term));
  }, [deployments.data, search]);

  const openCreate = () => setSearchParams((current) => {
    const next = new URLSearchParams(current);
    next.set('create', '1');
    return next;
  });
  const openPreset = (checkpoint: Checkpoint) => setSearchParams({
    create: '1',
    checkpoint: checkpoint.path,
    ...(checkpoint.combination_id ? { combination_id: checkpoint.combination_id } : {}),
    name: checkpoint.name_i18n?.[language] ?? checkpoint.experiment_name ?? checkpoint.path.split('/').slice(-1)[0] ?? '',
  });
  const closeCreate = () => setSearchParams((current) => {
    const next = new URLSearchParams(current);
    next.delete('create');
    next.delete('checkpoint_id');
    next.delete('checkpoint');
    next.delete('combination_id');
    next.delete('name');
    return next;
  }, { replace: true });

  return (
    <div className="page deployments-page">
      <PageIntro
        title={t('deployment.title')}
        subtitle={t('deployment.subtitle')}
        actions={<Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>{t('deployment.create')}</Button>}
      />
      {localPresets.length ? (
        <Card className="preset-section" title={t('deployment.localPresetsTitle')}>
          <div className="preset-card-grid compact">
            {localPresets.map((checkpoint) => (
              <Card key={checkpoint.id} size="small" className="preset-card">
                <Space wrap><Tag color="geekblue">{t('checkpoints.localPreset')}</Tag><Tag color="green">{checkpoint.combination_id}</Tag>{checkpoint.checkpoint_format_label_i18n?.[language] ? <Tag color={checkpoint.checkpoint_format === 'lerobot' ? 'cyan' : checkpoint.checkpoint_format === 'openpi' ? 'blue' : 'purple'}>{t('checkpoints.formatSource')} · {checkpoint.checkpoint_format_label_i18n[language]}</Tag> : null}</Space>
                <Typography.Title level={5}>{checkpoint.name_i18n?.[language] ?? checkpoint.experiment_name}</Typography.Title>
                <Typography.Paragraph type="secondary">{checkpoint.description_i18n?.[language] ?? checkpoint.description}</Typography.Paragraph>
                {(checkpoint.missing_requirements_i18n?.[language]?.length ?? 0) > 0 ? <Alert type="warning" showIcon message={t('deployment.presetNeedsParameters')} description={checkpoint.missing_requirements_i18n?.[language]?.join('、')} /> : null}
                <Button block type="primary" icon={<CloudServerOutlined />} onClick={() => openPreset(checkpoint)}>{t('deployment.usePreset')}</Button>
              </Card>
            ))}
          </div>
        </Card>
      ) : null}
      <Card>
        <div className="table-toolbar">
          <Input className="table-search" allowClear prefix={<SearchOutlined />} value={search} onChange={(event) => setSearch(event.target.value)} placeholder={t('common.search')} />
          <Button icon={<ExperimentOutlined />} onClick={() => void deployments.refetch()}>{t('common.refresh')}</Button>
        </div>
        <AsyncState loading={deployments.isLoading} error={deployments.error} empty={!rows.length} onRetry={() => void deployments.refetch()}>
          <Table<ModelDeployment>
            rowKey="id"
            dataSource={rows}
            scroll={{ x: 1100 }}
            onRow={(row) => ({ onClick: () => navigate(`/deployments/${row.id}`) })}
            columns={[
              { title: t('deployment.name'), dataIndex: 'name', fixed: 'left', render: (value: string, row) => <div><Typography.Text strong>{value}</Typography.Text><small className="table-subtitle">{row.owner_name ?? '—'}</small></div> },
              { title: t('common.status'), dataIndex: 'status', render: (status) => <StatusTag status={status} /> },
              { title: t('deployment.modelCombination'), dataIndex: 'combination_id', render: (value: string, row) => <div><Typography.Text>{value || '—'}</Typography.Text><small className="table-subtitle">{[row.backbone_id, row.action_head_id].filter(Boolean).join(' + ')}</small></div> },
              { title: t('deployment.gpuResources'), render: (_, row) => row.assigned_gpu_ids.length ? row.assigned_gpu_ids.map((id) => <span className="gpu-chip" key={id}>{id}</span>) : row.queue_position ? `#${row.queue_position}` : `${row.requested_gpu_count} GPU` },
              {
                title: t('deployment.endpoint'), width: 260,
                render: (_, row) => {
                  const endpoint = deploymentEndpoint(row);
                  return endpoint ? <Space align="start" onClick={(event) => event.stopPropagation()}><Typography.Text code className="table-wrap-text">{endpoint}</Typography.Text><Button type="text" size="small" icon={<CopyOutlined />} onClick={() => { void navigator.clipboard.writeText(endpoint); message.success(t('common.copied')); }} /></Space> : '—';
                },
              },
              { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm') : '—' },
              { title: t('common.actions'), fixed: 'right', render: (_, row) => <Button size="small" onClick={(event) => { event.stopPropagation(); navigate(`/deployments/${row.id}`); }}>{t('common.view')}</Button> },
            ]}
          />
        </AsyncState>
      </Card>
      {createOpen ? (
        <CreateDeploymentModal
          open
          onClose={closeCreate}
          onCreated={async (result) => {
            closeCreate();
            await queryClient.invalidateQueries({ queryKey: ['deployments'] });
            if (result.api_key) setSecret(result);
            else navigate(`/deployments/${result.deployment.id}`);
          }}
        />
      ) : null}
      <DeploymentApiKeyModal
        open={Boolean(secret?.api_key)}
        deployment={secret?.deployment}
        apiKey={secret?.api_key}
        onClose={() => {
          const id = secret?.deployment.id;
          setSecret(undefined);
          if (id) navigate(`/deployments/${id}`);
        }}
      />
    </div>
  );
}
