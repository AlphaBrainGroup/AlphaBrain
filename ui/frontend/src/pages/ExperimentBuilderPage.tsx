import {
  ArrowLeftOutlined,
  ArrowRightOutlined,
  CheckCircleFilled,
  CloudUploadOutlined,
  CodeOutlined,
  ExperimentOutlined,
  FileDoneOutlined,
  RocketOutlined,
  SaveOutlined,
  WarningFilled,
} from '@ant-design/icons';
import Editor from '@monaco-editor/react';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Alert,
  AutoComplete,
  Button,
  Card,
  Checkbox,
  Col,
  Collapse,
  Descriptions,
  Divider,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Radio,
  Result,
  Row,
  Select,
  Space,
  Steps,
  Switch,
  Tag,
  Typography,
  message,
} from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router-dom';
import YAML from 'yaml';
import { api, healthyGpus, templateDuplicateDetail } from '../api/client';
import type {
  Capability,
  CapabilityKind,
  DatasetValidationResult,
  ExperimentSpec,
  ParameterSchema,
  PreflightItem,
  PreflightResult,
  ResolveResult,
  TemplateDuplicateDetail,
  TemplateDuplicateMatch,
  WandbCategoryCapability,
  WandbRunConfig,
  WandbUploadCategory,
  WorkflowField,
} from '../api/types';
import { usePreferences } from '../app-context';
import { AsyncState } from '../components/AsyncState';
import { DatasetDirectoryPicker } from '../components/DatasetDirectoryPicker';
import { PageIntro } from '../components/PageIntro';
import { TrainingWorkflowFields } from '../components/TrainingWorkflowFields';
import { TemplateDuplicateModal } from '../components/TemplateDuplicateModal';
import { useGpuRefreshInterval } from '../hooks/useGpuRefreshInterval';
import { builderGpuIds, capabilityMatches, defaultWandbRunConfig, filterSupportedWandbCategories, isBuilderStepComplete, normalizeBuilderResources, workflowFieldActive } from './builder-validation';

interface BuilderValues {
  name: string;
  description?: string;
  backbone_id?: string;
  action_head_id?: string;
  method_id?: string;
  dataset_id?: string;
  dataset_source?: 'builtin' | 'registered' | 'mixture' | 'local';
  dataset_registration_id?: string;
  dataset_mixture_id?: string;
  use_local_dataset?: boolean;
  dataset_path?: string;
  batch_size: number;
  learning_rate: number;
  max_steps: number;
  save_interval: number;
  seed: number;
  num_workers: number;
  resource_strategy: 'auto' | 'fixed';
  gpu_count: number;
  gpu_ids?: number[];
  checkpoint_path?: string;
  resume_mode?: 'none' | 'weights_only' | 'full_state';
  resume_checkpoint?: string;
  module_parameters?: Record<string, unknown>;
  workflow_config?: Record<string, unknown>;
  wandb?: WandbRunConfig;
}

function named(capability: Capability, language: string): string {
  return language === 'zh-CN' && capability.name_zh ? capability.name_zh : capability.name;
}

function localizedWandbText(
  value: Partial<Record<'zh-CN' | 'en-US', string>> | undefined,
  language: string,
  fallback: string,
): string {
  return value?.[language === 'en-US' ? 'en-US' : 'zh-CN'] ?? value?.['en-US'] ?? value?.['zh-CN'] ?? fallback;
}

function flattenCapabilities(data: Awaited<ReturnType<typeof api.capabilities>> | undefined): Capability[] {
  if (!data) return [];
  if (data.capabilities?.length) return data.capabilities;
  const withKind = (values: Capability[] | undefined, kind: CapabilityKind) =>
    (values ?? []).map((item) => ({ ...item, kind: item.kind ?? kind }));
  return [
    ...withKind(data.backbones, 'backbone'),
    ...withKind(data.action_heads, 'action_head'),
    ...withKind(data.methods, 'method'),
    ...withKind(data.datasets, 'dataset'),
  ];
}

const CORE_PARAMETER_KEYS = new Set([
  'batch_size',
  'learning_rate',
  'max_steps',
  'save_interval',
  'seed',
  'num_workers',
]);
const DEDICATED_PARAMETER_KEYS = new Set(['pretrained_checkpoint']);

function parseOverrides(source: string, invalidMessage: string): Record<string, unknown> {
  if (!source.trim()) return {};
  const parsed: unknown = YAML.parse(source);
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error(invalidMessage);
  }
  return parsed as Record<string, unknown>;
}

function DynamicField({ schema, language }: { schema: ParameterSchema; language: string }) {
  const input = (() => {
    if (schema.type === 'boolean') return <Checkbox />;
    if (schema.type === 'select') return <Select options={schema.options} />;
    if (['number', 'integer'].includes(schema.type)) {
      return <InputNumber min={schema.min} max={schema.max} precision={schema.type === 'integer' ? 0 : undefined} className="full-width" />;
    }
    return <Input />;
  })();
  return (
    <Form.Item
      name={['module_parameters', schema.key]}
      label={language === 'zh-CN' ? (schema.label_zh ?? schema.label ?? schema.key) : (schema.label ?? schema.key)}
      tooltip={schema.description}
      valuePropName={schema.type === 'boolean' ? 'checked' : 'value'}
      rules={[{ required: schema.required }]}
      initialValue={schema.default}
    >
      {input}
    </Form.Item>
  );
}

export function ExperimentBuilderPage() {
  const { t } = useTranslation();
  const { language, theme } = usePreferences();
  const gpuRefreshInterval = useGpuRefreshInterval();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const templateId = searchParams.get('template');
  const checkpointPath = searchParams.get('checkpoint');
  const resumeModeParam = searchParams.get('resume_mode');
  const initialResumeMode = resumeModeParam === 'weights_only' || resumeModeParam === 'full_state' ? resumeModeParam : 'none';
  const [form] = Form.useForm<BuilderValues>();
  const [step, setStep] = useState(0);
  const [expertYaml, setExpertYaml] = useState(() => `# ${t('builder.expertOverrides')}\n{}\n`);
  const [templateParameters, setTemplateParameters] = useState<Record<string, unknown>>({});
  const [resolved, setResolved] = useState<ResolveResult>();
  const [preflight, setPreflight] = useState<PreflightResult>();
  const [datasetValidation, setDatasetValidation] = useState<DatasetValidationResult>();
  const [pendingTemplateDuplicate, setPendingTemplateDuplicate] = useState<{
    detail: TemplateDuplicateDetail;
    spec: ExperimentSpec;
  }>();

  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const includeExperimental = Boolean(me.data?.experimental_enabled);
  const capabilityQuery = useQuery({
    queryKey: ['capabilities', includeExperimental],
    queryFn: () => api.capabilities(includeExperimental),
  });
  const templateQuery = useQuery({
    queryKey: ['template', templateId],
    queryFn: () => api.templates.get(templateId!),
    enabled: Boolean(templateId),
  });
  const checkpointQuery = useQuery({ queryKey: ['checkpoints'], queryFn: api.checkpoints });
  const registeredDatasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: api.datasets.list });
  const datasetMixturesQuery = useQuery({ queryKey: ['dataset-mixtures'], queryFn: api.datasets.mixtures.list });
  const trainingTargetQuery = useQuery({ queryKey: ['training-target'], queryFn: api.trainingTarget });
  const gpuQuery = useQuery({ queryKey: ['gpus'], queryFn: api.gpus, refetchInterval: gpuRefreshInterval });
  const localGpus = healthyGpus(gpuQuery.data);
  const remoteTraining = trainingTargetQuery.data?.mode === 'remote';
  const selectableGpuIds = trainingTargetQuery.isSuccess
    ? builderGpuIds(localGpus.map((gpu) => gpu.index), trainingTargetQuery.data)
    : [];
  const selectableGpus = remoteTraining
    ? selectableGpuIds.map((index) => ({
        index,
        name: t('builder.remoteGpu'),
        available: true,
      }))
    : localGpus;
  const detectedGpuCount = selectableGpus.length;
  const singleGpu = detectedGpuCount === 1;

  useEffect(() => {
    if (!templateQuery.data?.spec) return;
    const spec = templateQuery.data.spec;
    const moduleParameters = Object.fromEntries(
      Object.entries(spec.parameters).filter(([key]) => !CORE_PARAMETER_KEYS.has(key)),
    );
    setTemplateParameters(moduleParameters);
    form.setFieldsValue({
      name: t('builder.copyName', { name: templateQuery.data.name }),
      description: spec.description,
      backbone_id: spec.backbone_id,
      action_head_id: spec.action_head_id,
      method_id: spec.method_id,
      dataset_id: spec.dataset_id,
      dataset_source: spec.dataset_registration_id ? 'registered' : spec.dataset_mixture_id ? 'mixture' : spec.dataset_path ? 'local' : 'builtin',
      dataset_registration_id: spec.dataset_registration_id,
      dataset_mixture_id: spec.dataset_mixture_id,
      use_local_dataset: Boolean(spec.dataset_path),
      dataset_path: spec.dataset_path,
      batch_size: Number(spec.parameters.batch_size ?? 8),
      learning_rate: Number(spec.parameters.learning_rate ?? 0.0001),
      max_steps: Number(spec.parameters.max_steps ?? 10000),
      save_interval: Number(spec.parameters.save_interval ?? 1000),
      seed: Number(spec.parameters.seed ?? 42),
      num_workers: Number(spec.parameters.num_workers ?? 4),
      resource_strategy: spec.resources.strategy,
      gpu_count: spec.resources.gpu_count,
      gpu_ids: spec.resources.gpu_ids,
      checkpoint_path: spec.checkpoint_path,
      workflow_config: spec.workflow?.config ?? {},
      resume_mode: spec.resume?.mode ?? 'none',
      resume_checkpoint: spec.resume?.checkpoint,
      module_parameters: moduleParameters,
      wandb: spec.wandb ?? defaultWandbRunConfig(),
    });
    setExpertYaml(YAML.stringify(spec.expert_overrides ?? {}));
  }, [form, templateQuery.data, t]);

  useEffect(() => {
    if (!singleGpu) return;
    form.setFieldsValue({ resource_strategy: 'auto', gpu_count: 1, gpu_ids: undefined });
  }, [form, singleGpu, templateQuery.data]);

  const all = useMemo(() => flattenCapabilities(capabilityQuery.data), [capabilityQuery.data]);
  const backboneId = Form.useWatch('backbone_id', form);
  const actionHeadId = Form.useWatch('action_head_id', form);
  const methodId = Form.useWatch('method_id', form);
  const datasetId = Form.useWatch('dataset_id', form);
  const datasetSource = Form.useWatch('dataset_source', form) ?? 'builtin';
  const datasetRegistrationId = Form.useWatch('dataset_registration_id', form);
  const datasetMixtureId = Form.useWatch('dataset_mixture_id', form);
  const datasetPath = Form.useWatch('dataset_path', form);
  const resourceStrategy = Form.useWatch('resource_strategy', form);
  const workflowConfig = (Form.useWatch('workflow_config', form) ?? {}) as Record<string, unknown>;
  const resumeMode = Form.useWatch('resume_mode', form) ?? 'none';
  const wandbEnabled = Boolean(Form.useWatch(['wandb', 'enabled'], form));
  const wandbMode = Form.useWatch(['wandb', 'mode'], form);
  const selectedWandbCategories = Form.useWatch(['wandb', 'categories'], form) as WandbUploadCategory[] | undefined;
  const builderValues = Form.useWatch([], form) ?? {};
  const selectedIds = [backboneId, actionHeadId, methodId, datasetId];
  const selected = all.filter((item) => selectedIds.includes(item.id));
  const selectedCombination = capabilityQuery.data?.combinations?.find((combo) =>
    combo.backbone === backboneId
    && combo.action_head === actionHeadId
    && combo.method === methodId
    && combo.datasets.includes(selectedIds[3] ?? ''),
  );
  const selectedWorkflow = selectedCombination?.workflow;
  useEffect(() => {
    if (!selectedWorkflow) return;
    const defaults = Object.fromEntries(
      selectedWorkflow.fields.flatMap((field) => field.default === undefined ? [] : [[field.key, field.default]]),
    );
    const current = form.getFieldValue('workflow_config') as Record<string, unknown> | undefined;
    form.setFieldValue('workflow_config', { ...defaults, ...(current ?? {}) });
  }, [form, selectedCombination?.id, selectedWorkflow?.id]);
  const activeWorkflowFields = (selectedWorkflow?.fields ?? []).filter((field: WorkflowField) => workflowFieldActive(
    field,
    workflowConfig,
    { method: methodId, algorithm: selectedCombination?.rl_algorithm },
  ));
  const wandbCategories = selectedCombination?.wandb_categories ?? [];
  const supportedWandbCategoryIds = useMemo(
    () => wandbCategories.filter((item) => item.supported).map((item) => item.id),
    [wandbCategories],
  );
  useEffect(() => {
    if (!wandbCategories.length) return;
    const current = form.getFieldValue(['wandb', 'categories']) as WandbUploadCategory[] | undefined;
    if (!current) return;
    const filtered = filterSupportedWandbCategories(current, supportedWandbCategoryIds) as WandbUploadCategory[];
    if (filtered.length !== current.length) form.setFieldValue(['wandb', 'categories'], filtered);
  }, [form, supportedWandbCategoryIds, wandbCategories.length]);
  const selectedMethod = all.find((item) => item.id === methodId && item.kind === 'method');
  const hasExperimental = selected.some((item) => item.status === 'experimental') || selectedCombination?.status === 'experimental';
  const params = Array.from(
    new Map(selected.flatMap((item) => item.parameters ?? []).map((item) => [item.key, item])).values(),
  ).filter((item) => !DEDICATED_PARAMETER_KEYS.has(item.key));
  const options = (kind: CapabilityKind) => {
    const combinations = capabilityQuery.data?.combinations ?? [];
    const reachable = combinations.filter((combo) => {
      if (kind !== 'backbone' && backboneId && combo.backbone !== backboneId) return false;
      if (!['backbone', 'action_head'].includes(kind) && actionHeadId && combo.action_head !== actionHeadId) return false;
      if (kind === 'dataset' && methodId && combo.method !== methodId) return false;
      return true;
    });
    const allowed = new Set(reachable.flatMap((combo) => {
      if (kind === 'backbone') return [combo.backbone];
      if (kind === 'action_head') return [combo.action_head];
      if (kind === 'method') return [combo.method];
      return combo.datasets;
    }));
    return all
      .filter((item) => item.kind === kind && (!combinations.length || allowed.has(item.id)))
      .map((item) => ({
        value: item.id,
        searchText: [item.id, item.name, item.name_zh, item.pretrained_directory?.path].filter(Boolean).join(' '),
        label: (
          <Space>
            <span>{named(item, language)}</span>
            {item.status === 'experimental' ? <Tag color="orange">{t('builder.experimental')}</Tag> : null}
            {kind === 'backbone' && item.pretrained_directory?.required ? (
              <Tag color={item.pretrained_directory.configured && item.pretrained_directory.exists ? 'green' : 'red'}>
                {item.pretrained_directory.configured && item.pretrained_directory.exists
                  ? t('builder.modelDirectoryReady')
                  : item.pretrained_directory.configured
                    ? t('builder.modelDirectoryMissing')
                    : t('builder.modelDirectoryUnconfigured')}
              </Tag>
            ) : null}
            {kind === 'backbone' && item.pretrained_directory && !item.pretrained_directory.required ? (
              <Tag>{t('builder.modelDirectoryNotRequired')}</Tag>
            ) : null}
          </Space>
        ),
      }));
  };
  const filterCapabilityOption = (input: string, option?: { value?: string; searchText?: string }) =>
    capabilityMatches(input, [option?.value, option?.searchText]);
  const expertYamlValid = useMemo(() => {
    try {
      parseOverrides(expertYaml, t('builder.invalidOverrides'));
      return true;
    } catch {
      return false;
    }
  }, [expertYaml, t]);
  const localDatasetValid = Boolean(
    datasetValidation?.valid
    && datasetValidation.dataset_id === datasetId
    && datasetValidation.normalized_root === datasetPath,
  );
  const datasetReferenceReady = datasetSource === 'builtin'
    || (datasetSource === 'local' && localDatasetValid)
    || (datasetSource === 'registered' && Boolean(datasetRegistrationId))
    || (datasetSource === 'mixture' && Boolean(datasetMixtureId));
  const completionOptions = {
    localDatasetValid,
    datasetReferenceReady,
    requiresCheckpoint: Boolean(selectedMethod?.requires_checkpoint),
    requiredParameterKeys: params.filter((item) => item.required).map((item) => item.key),
    requiredWorkflowKeys: activeWorkflowFields.filter((item) => item.required).map((item) => item.key),
    expertYamlValid,
    singleGpu,
    supportedWandbCategoryIds,
  };
  const stepComplete = [0, 1, 2, 3].map((index) => isBuilderStepComplete(index, builderValues, completionOptions));
  const stepAccessible = [true, stepComplete[0], stepComplete[0] && stepComplete[1], stepComplete[0] && stepComplete[1] && stepComplete[2]];

  const buildSpec = async (): Promise<ExperimentSpec> => {
    const values = await form.validateFields();
    const rawWandb = values.wandb ?? defaultWandbRunConfig();
    const rawTags = Array.isArray(rawWandb.tags) ? rawWandb.tags : [];
    const rawCategories = Array.isArray(rawWandb.categories) ? rawWandb.categories : [];
    const wandb: WandbRunConfig = {
      enabled: Boolean(rawWandb.enabled),
      mode: rawWandb.mode === 'offline' || rawWandb.mode === 'disabled' ? rawWandb.mode : 'online',
      project: (rawWandb.project ?? 'AlphaBrain').trim(),
      entity: (rawWandb.entity ?? '').trim(),
      run_name: (rawWandb.run_name ?? '').trim(),
      group: (rawWandb.group ?? '').trim(),
      job_type: (rawWandb.job_type ?? '').trim(),
      tags: Array.from(new Set(rawTags.map((item) => String(item).trim()).filter(Boolean))),
      notes: rawWandb.notes ?? '',
      categories: filterSupportedWandbCategories(rawCategories, supportedWandbCategoryIds) as WandbUploadCategory[],
    };
    return {
      spec_version: 2,
      name: values.name,
      description: values.description,
      template_id: templateId ?? undefined,
      backbone_id: values.backbone_id,
      action_head_id: values.action_head_id,
      method_id: values.method_id,
      dataset_id: values.dataset_id,
      dataset_path: values.dataset_source === 'local' ? values.dataset_path : undefined,
      dataset_registration_id: values.dataset_source === 'registered' ? values.dataset_registration_id : undefined,
      dataset_mixture_id: values.dataset_source === 'mixture' ? values.dataset_mixture_id : undefined,
      checkpoint_path: values.checkpoint_path || (initialResumeMode === 'none' ? checkpointPath : undefined) || undefined,
      workflow: selectedWorkflow ? {
        id: selectedWorkflow.id,
        schema_version: selectedWorkflow.schema_version,
        config: values.workflow_config ?? {},
      } : undefined,
      resume: {
        mode: values.resume_mode ?? 'none',
        ...((values.resume_mode ?? 'none') !== 'none' && values.resume_checkpoint
          ? { checkpoint: values.resume_checkpoint }
          : {}),
      },
      parameters: {
        batch_size: values.batch_size,
        learning_rate: values.learning_rate,
        max_steps: values.max_steps,
        save_interval: values.save_interval,
        seed: values.seed,
        num_workers: values.num_workers,
        ...templateParameters,
        ...(values.module_parameters ?? {}),
      },
      resources: normalizeBuilderResources(values, detectedGpuCount),
      wandb,
      expert_overrides: parseOverrides(expertYaml, t('builder.invalidOverrides')),
      experimental_confirmed: hasExperimental,
    };
  };

  const resolveMutation = useMutation({
    mutationFn: async () => api.experiments.resolve(await buildSpec()),
    onSuccess: setResolved,
    onError: (error) => message.error(error instanceof Error ? error.message : String(error)),
  });
  const preflightMutation = useMutation({
    mutationFn: async () => api.experiments.preflight(await buildSpec()),
    onSuccess: setPreflight,
    onError: (error) => message.error(error instanceof Error ? error.message : String(error)),
  });
  const submitMutation = useMutation({
    mutationFn: async () => api.experiments.submit(await buildSpec()),
    onSuccess: (result) => {
      message.success(t('builder.success'));
      const jobId = result.jobs[0]?.id ?? result.experiment.job_id;
      navigate(jobId ? `/jobs/${jobId}` : `/experiments/${result.experiment.id}`);
    },
    onError: (error) => message.error(error instanceof Error ? error.message : String(error)),
  });
  const saveTemplateMutation = useMutation({
    mutationFn: async ({ spec, allowDuplicate }: { spec: ExperimentSpec; allowDuplicate: boolean }) => {
      return api.templates.create(
        { name: spec.name, description: spec.description, visibility: 'personal', spec },
        allowDuplicate,
      );
    },
    onSuccess: () => {
      setPendingTemplateDuplicate(undefined);
      message.success(t('builder.templateSaved'));
    },
    onError: (error, variables) => {
      const detail = templateDuplicateDetail(error);
      if (detail) setPendingTemplateDuplicate({ detail, spec: variables.spec });
      else message.error(error instanceof Error ? error.message : String(error));
    },
  });

  const saveTemplate = async () => {
    try {
      const spec = await buildSpec();
      saveTemplateMutation.mutate({ spec, allowDuplicate: false });
    } catch (error) {
      message.error(error instanceof Error ? error.message : String(error));
    }
  };

  const openDuplicateTemplate = (match: TemplateDuplicateMatch) => {
    setPendingTemplateDuplicate(undefined);
    navigate(`/experiments/new?template=${encodeURIComponent(match.id)}`);
  };

  const validateAndAdvance = async () => {
    try {
      if (step === 0) {
        await form.validateFields([
          'backbone_id', 'action_head_id', 'method_id', 'dataset_id', 'dataset_source',
          ...(datasetSource === 'local' ? ['dataset_path'] : []),
          ...(datasetSource === 'registered' ? ['dataset_registration_id'] : []),
          ...(datasetSource === 'mixture' ? ['dataset_mixture_id'] : []),
        ]);
      } else if (step === 1) {
        await form.validateFields();
      }
      if (step === 2) parseOverrides(expertYaml, t('builder.invalidOverrides'));
      setStep((value) => Math.min(3, value + 1));
    } catch (error) {
      if (error instanceof Error) message.error(error.message);
    }
  };

  const submit = () => {
    const proceed = () => submitMutation.mutate();
    if (hasExperimental) {
      Modal.confirm({
        title: t('builder.experimental'),
        icon: <WarningFilled style={{ color: '#d97706' }} />,
        content: t('builder.experimentalRisk'),
        okText: t('common.confirm'),
        okButtonProps: { danger: true },
        onOk: proceed,
      });
    } else proceed();
  };

  const renderCapability = (capability?: Capability) => {
    if (!capability) return null;
    const directory = capability.kind === 'backbone' ? capability.pretrained_directory : undefined;
    const directoryReady = Boolean(directory?.configured && directory.exists);
    const directoryIssue = directory?.issue?.message_i18n?.[language];
    const directoryLabel = !directory
      ? undefined
      : !directory.required
        ? t('builder.modelDirectoryNotRequired')
        : directoryReady
          ? t('builder.modelDirectoryReady')
          : directory.configured
            ? t('builder.modelDirectoryMissing')
            : t('builder.modelDirectoryUnconfigured');
    return (
      <div className={`selected-capability${directory?.required && !directoryReady ? ' directory-warning' : ''}`}>
        <span className="capability-icon"><ExperimentOutlined /></span>
        <div className="capability-summary">
          <b>{named(capability, language)}</b>
          <small>{language === 'zh-CN' ? (capability.description_zh ?? capability.description) : capability.description}</small>
          {directory?.path ? <small className="pretrained-directory-path" title={directory.path}>{directory.path}</small> : null}
        </div>
        <Space direction="vertical" size={2} align="end">
          <Tag color={capability.status === 'experimental' ? 'orange' : capability.status === 'verified' ? 'green' : 'default'}>
            {t(`capability.${capability.status}`)}
          </Tag>
          {directoryLabel ? <Tag color={directory?.required ? (directoryReady ? 'green' : 'red') : 'default'}>{directoryLabel}</Tag> : null}
        </Space>
        {directory?.required && !directoryReady ? (
          <div className="pretrained-directory-issue">
            <WarningFilled />
            <span>{directoryIssue ?? t('builder.modelDirectoryIssue')}</span>
          </div>
        ) : null}
      </div>
    );
  };

  const issueMessage = (item: PreflightItem): string => {
    const localized = item.message_i18n?.[language];
    if (localized) return localized;
    return t(`preflight.${item.id ?? 'unknown'}`, {
      defaultValue: item.message ?? item.title,
      path: item.path ?? '',
    });
  };

  return (
    <div className="page builder-page">
      <PageIntro title={t('builder.title')} subtitle={t('builder.subtitle')} />
      <Card className="builder-shell">
        <Steps
          current={step}
          onChange={(next) => { if (next <= step || stepAccessible[next]) setStep(next); }}
          items={[
            { title: t('builder.architecture'), icon: <ExperimentOutlined /> },
            { title: t('builder.parameters'), icon: <FileDoneOutlined />, disabled: !stepAccessible[1] },
            { title: t('builder.expert'), icon: <CodeOutlined />, disabled: !stepAccessible[2] },
            { title: t('builder.review'), icon: <RocketOutlined />, disabled: !stepAccessible[3] },
          ]}
        />
        <Divider />
        <AsyncState
          loading={capabilityQuery.isLoading || trainingTargetQuery.isLoading || (!remoteTraining && gpuQuery.isLoading)}
          error={capabilityQuery.error ?? trainingTargetQuery.error ?? (!remoteTraining ? gpuQuery.error : null)}
          onRetry={() => void Promise.all([capabilityQuery.refetch(), trainingTargetQuery.refetch(), gpuQuery.refetch()])}
        >
          <Form<BuilderValues>
            form={form}
            layout="vertical"
            onValuesChange={(changed) => {
              setResolved(undefined);
              setPreflight(undefined);
              if ('dataset_id' in changed || 'dataset_path' in changed || 'dataset_source' in changed) setDatasetValidation(undefined);
            }}
            initialValues={{
              batch_size: 8,
              learning_rate: 0.0001,
              max_steps: 10000,
              save_interval: 1000,
              seed: 42,
              num_workers: 4,
              resource_strategy: 'auto',
              gpu_count: 1,
              wandb: defaultWandbRunConfig(),
              use_local_dataset: false,
              dataset_source: 'builtin',
              checkpoint_path: initialResumeMode === 'none' ? checkpointPath ?? undefined : undefined,
              resume_mode: initialResumeMode,
              resume_checkpoint: initialResumeMode !== 'none' ? checkpointPath ?? undefined : undefined,
              workflow_config: {},
            }}
          >
            <div hidden={step !== 0} className="builder-step">
              {includeExperimental ? <Alert className="step-alert" type="warning" showIcon message={t('builder.experimentalRisk')} /> : null}
              <Row gutter={[22, 18]}>
                <Col xs={24} lg={12}>
                  <Form.Item name="backbone_id" label={`1 · ${t('builder.backbone')}`} rules={[{ required: true }]}>
                    <Select
                      size="large"
                      showSearch
                      filterOption={filterCapabilityOption}
                      options={options('backbone')}
                      onChange={() => { form.setFieldsValue({ action_head_id: undefined, method_id: undefined, dataset_id: undefined, workflow_config: undefined }); setDatasetValidation(undefined); }}
                    />
                  </Form.Item>
                  {renderCapability(all.find((item) => item.id === backboneId))}
                </Col>
                <Col xs={24} lg={12}>
                  <Form.Item name="action_head_id" label={`2 · ${t('builder.actionHead')}`} rules={[{ required: true }]}>
                    <Select
                      size="large"
                      disabled={!backboneId}
                      placeholder={!backboneId ? t('builder.selectPrevious') : undefined}
                      showSearch
                      filterOption={filterCapabilityOption}
                      options={options('action_head')}
                      onChange={() => form.setFieldsValue({ method_id: undefined, dataset_id: undefined, workflow_config: undefined })}
                    />
                  </Form.Item>
                  {renderCapability(all.find((item) => item.id === actionHeadId))}
                </Col>
                <Col xs={24} lg={12}>
                  <Form.Item name="method_id" label={`3 · ${t('builder.method')}`} rules={[{ required: true }]}>
                    <Select
                      size="large"
                      disabled={!actionHeadId}
                      placeholder={!actionHeadId ? t('builder.selectPrevious') : undefined}
                      showSearch
                      filterOption={filterCapabilityOption}
                      options={options('method')}
                      onChange={() => form.setFieldsValue({ dataset_id: undefined, workflow_config: undefined })}
                    />
                  </Form.Item>
                  {renderCapability(all.find((item) => item.id === methodId))}
                </Col>
                <Col xs={24} lg={12}>
                  <Form.Item name="dataset_id" label={`4 · ${t('builder.dataset')}`} rules={[{ required: true }]}>
                    <Select
                      size="large"
                      disabled={!methodId}
                      placeholder={!methodId ? t('builder.selectPrevious') : undefined}
                      showSearch
                      filterOption={filterCapabilityOption}
                      options={options('dataset')}
                      onChange={(datasetId) => {
                        form.setFieldValue('workflow_config', undefined);
                        setDatasetValidation(undefined);
                        const combo = capabilityQuery.data?.combinations?.find((item) => item.backbone === backboneId && item.action_head === actionHeadId && item.method === methodId && item.datasets.includes(datasetId));
                        if (combo?.min_gpus && !singleGpu) form.setFieldValue('gpu_count', combo.min_gpus);
                      }}
                    />
                  </Form.Item>
                  {renderCapability(all.find((item) => item.id === selectedIds[3]))}
                </Col>
                <Col span={24}>
                  <Card size="small" title={t('builder.localDataset')}>
                    <Form.Item name="dataset_source" label={t('builder.datasetSource')} rules={[{ required: true }]}>
                      <Radio.Group
                        disabled={!datasetId}
                        optionType="button"
                        buttonStyle="solid"
                        options={[
                          { label: t('builder.builtinDataset'), value: 'builtin' },
                          { label: t('builder.registeredDataset'), value: 'registered' },
                          { label: t('builder.datasetMixture'), value: 'mixture' },
                          { label: t('builder.localDirectory'), value: 'local' },
                        ]}
                        onChange={(event) => {
                          const source = event.target.value as BuilderValues['dataset_source'];
                          form.setFieldsValue({
                            use_local_dataset: source === 'local',
                            dataset_path: undefined,
                            dataset_registration_id: undefined,
                            dataset_mixture_id: undefined,
                          });
                          setDatasetValidation(undefined);
                        }}
                      />
                    </Form.Item>
                    <Form.Item name="use_local_dataset" hidden><Input /></Form.Item>
                    {datasetSource === 'registered' ? (
                      <Form.Item name="dataset_registration_id" label={t('builder.registeredDataset')} rules={[{ required: true }]} extra={t('builder.registeredDatasetHint')}>
                        <Select
                          showSearch
                          loading={registeredDatasetsQuery.isLoading}
                          optionFilterProp="label"
                          options={(registeredDatasetsQuery.data ?? [])
                            .filter((item) => item.status === 'ready')
                            .map((item) => ({ value: item.id, label: `${item.name} · ${item.format}`, title: item.path }))}
                          placeholder={t('builder.selectRegisteredDataset')}
                        />
                      </Form.Item>
                    ) : null}
                    {datasetSource === 'mixture' ? (
                      <Form.Item name="dataset_mixture_id" label={t('builder.datasetMixture')} rules={[{ required: true }]} extra={t('builder.datasetMixtureHint')}>
                        <Select
                          showSearch
                          loading={datasetMixturesQuery.isLoading}
                          optionFilterProp="label"
                          options={(datasetMixturesQuery.data ?? []).map((item) => ({ value: item.id, label: `${item.name} · v${item.version}`, title: item.description }))}
                          placeholder={t('builder.selectDatasetMixture')}
                        />
                      </Form.Item>
                    ) : null}
                    {datasetSource === 'local' ? (
                      <Form.Item name="dataset_path" label={t('builder.datasetPath')} rules={[{ required: true }]}>
                        <DatasetDirectoryPicker
                          datasetId={datasetId}
                          language={language}
                          validation={datasetValidation}
                          onValidated={setDatasetValidation}
                        />
                      </Form.Item>
                    ) : <Typography.Paragraph type="secondary">{datasetSource === 'builtin' ? t('builder.builtinDatasetHint') : null}</Typography.Paragraph>}
                  </Card>
                </Col>
              </Row>
            </div>

            <div hidden={step !== 1} className="wandb-run-wrap">
              <Card
                size="small"
                className="wandb-run-card"
                title={<Space><CloudUploadOutlined />{t('builder.wandbTitle')}</Space>}
                extra={(
                  <Space>
                    <Typography.Text type="secondary">{t(wandbEnabled ? 'common.enabled' : 'common.disabled')}</Typography.Text>
                    <Form.Item name={['wandb', 'enabled']} valuePropName="checked" noStyle>
                      <Switch
                        disabled={!wandbCategories.length}
                        onChange={(enabled) => {
                          if (enabled && form.getFieldValue(['wandb', 'mode']) === 'disabled') {
                            form.setFieldValue(['wandb', 'mode'], 'online');
                          }
                        }}
                      />
                    </Form.Item>
                  </Space>
                )}
              >
                <Typography.Paragraph type="secondary">{t('builder.wandbDescription')}</Typography.Paragraph>
                {!wandbCategories.length ? (
                  <Alert type="warning" showIcon message={t('builder.wandbCapabilitiesUnavailable')} />
                ) : null}
                {wandbEnabled ? (
                  <>
                    <Alert
                      className="wandb-mode-notice"
                      type={wandbMode === 'offline' ? 'info' : 'warning'}
                      showIcon
                      message={t(wandbMode === 'offline' ? 'builder.wandbOfflineHint' : 'builder.wandbOnlineHint')}
                    />
                    <div className="form-grid-2">
                      <Form.Item name={['wandb', 'mode']} label={t('builder.wandbMode')} rules={[{ required: true }]}>
                        <Radio.Group
                          optionType="button"
                          buttonStyle="solid"
                          options={[
                            { label: t('builder.wandbOnline'), value: 'online' },
                            { label: t('builder.wandbOffline'), value: 'offline' },
                          ]}
                        />
                      </Form.Item>
                      <Form.Item name={['wandb', 'project']} label={t('builder.wandbProject')} rules={[{ required: true, whitespace: true, max: 128 }]}>
                        <Input placeholder="AlphaBrain" />
                      </Form.Item>
                      <Form.Item name={['wandb', 'entity']} label={t('builder.wandbEntity')} rules={[{ max: 128 }]}>
                        <Input placeholder={t('builder.wandbOptional')} />
                      </Form.Item>
                      <Form.Item name={['wandb', 'run_name']} label={t('builder.wandbRunName')} rules={[{ max: 256 }]} extra={t('builder.wandbRunNameHint')}>
                        <Input placeholder={t('builder.wandbRunNamePlaceholder')} />
                      </Form.Item>
                      <Form.Item name={['wandb', 'group']} label={t('builder.wandbGroup')} rules={[{ max: 256 }]}>
                        <Input placeholder={t('builder.wandbOptional')} />
                      </Form.Item>
                      <Form.Item name={['wandb', 'job_type']} label={t('builder.wandbJobType')} rules={[{ max: 128 }]}>
                        <Input placeholder={t('builder.wandbOptional')} />
                      </Form.Item>
                    </div>
                    <Form.Item
                      name={['wandb', 'tags']}
                      label={t('builder.wandbTags')}
                      rules={[{
                        validator: async (_, tags: string[] | undefined) => {
                          if ((tags ?? []).every((tag) => tag.trim().length > 0 && tag.length <= 128 && !tag.includes(','))) return;
                          throw new Error(t('builder.wandbTagInvalid'));
                        },
                      }]}
                    >
                      <Select mode="tags" tokenSeparators={[',']} placeholder={t('builder.wandbTagsPlaceholder')} />
                    </Form.Item>
                    <Typography.Title level={5} className="wandb-categories-title">{t('builder.wandbUploadContent')}</Typography.Title>
                    <Typography.Paragraph type="secondary">{t('builder.wandbUploadContentHint')}</Typography.Paragraph>
                    <Form.Item name={['wandb', 'categories']}>
                      <Checkbox.Group className="wandb-category-grid">
                        {wandbCategories.map((category: WandbCategoryCapability) => {
                          const title = localizedWandbText(category.title, language, category.id);
                          const description = localizedWandbText(category.description, language, '');
                          const reason = localizedWandbText(category.reason, language, t('builder.wandbCategoryUnsupported'));
                          return (
                            <div key={category.id} className={`wandb-category-option${category.supported ? '' : ' unsupported'}`}>
                              <Checkbox value={category.id} disabled={!category.supported}>
                                <b>{title}</b>
                              </Checkbox>
                              <small>{description}</small>
                              {!category.supported ? <Tag>{reason}</Tag> : null}
                            </div>
                          );
                        })}
                      </Checkbox.Group>
                    </Form.Item>
                    <Form.Item name={['wandb', 'notes']} label={t('builder.wandbNotes')} rules={[{ max: 10_000 }]}>
                      <Input.TextArea rows={3} showCount maxLength={10_000} placeholder={t('builder.wandbNotesPlaceholder')} />
                    </Form.Item>
                  </>
                ) : null}
              </Card>
            </div>

            <div hidden={step !== 1} className="builder-step">
              {selectedWorkflow?.fields.length ? (
                <div style={{ marginBottom: 18 }}>
                  <TrainingWorkflowFields
                    schema={selectedWorkflow}
                    fields={activeWorkflowFields}
                    language={language}
                  />
                </div>
              ) : null}
              <Row gutter={[28, 20]}>
                <Col xs={24} xl={16}>
                  <Card size="small" title={t('builder.parameters')}>
                    <Form.Item name="name" label={t('builder.experimentName')} rules={[{ required: true, min: 2 }]}>
                      <Input size="large" placeholder="qwen-oft-libero-001" />
                    </Form.Item>
                    <Form.Item name="description" label={t('builder.description')}><Input.TextArea rows={2} /></Form.Item>
                    {selectedMethod?.requires_checkpoint ? (
                      <Form.Item
                        name="checkpoint_path"
                        label={language === 'zh-CN' ? '训练起点 Checkpoint' : 'Training base checkpoint'}
                        extra={language === 'zh-CN'
                          ? 'R-STDP / RL 等方法需要的基础模型权重；它与“恢复完整训练状态”不是同一个概念。'
                          : 'Base model weights required by methods such as R-STDP and RL; this is separate from full-state resume.'}
                        rules={[{ required: true }]}
                      >
                        <AutoComplete
                          options={(checkpointQuery.data ?? []).map((item) => ({ value: item.path, label: `${item.experiment_name ?? item.experiment_id ?? t('checkpoints.experiment')} · ${t('checkpoints.step')} ${item.step ?? '—'}` }))}
                          placeholder="/path/to/checkpoint"
                          filterOption={(input, option) => String(option?.value ?? '').toLowerCase().includes(input.toLowerCase())}
                        />
                      </Form.Item>
                    ) : null}
                    <Divider titlePlacement="start" plain>
                      {language === 'zh-CN' ? 'Checkpoint 加载方式' : 'Checkpoint loading'}
                    </Divider>
                    <Form.Item name="resume_mode" label={language === 'zh-CN' ? '启动方式' : 'Start mode'}>
                      <Radio.Group
                        optionType="button"
                        buttonStyle="solid"
                        options={[
                          { label: language === 'zh-CN' ? '从头训练' : 'Fresh run', value: 'none' },
                          { label: language === 'zh-CN' ? '仅加载权重' : 'Weights only', value: 'weights_only' },
                          { label: language === 'zh-CN' ? '恢复完整状态' : 'Full state', value: 'full_state' },
                        ]}
                      />
                    </Form.Item>
                    <Alert
                      type={resumeMode === 'full_state' ? 'warning' : 'info'}
                      showIcon
                      message={resumeMode === 'full_state'
                        ? (language === 'zh-CN'
                          ? '恢复模型、优化器、学习率调度器与随机状态；通常要求 GPU 数量与原任务一致。'
                          : 'Restores model, optimizer, scheduler, and RNG state; the original GPU count is normally required.')
                        : resumeMode === 'weights_only'
                          ? (language === 'zh-CN'
                            ? '只加载模型权重，优化器和训练步数重新开始。'
                            : 'Loads model weights only; optimizer and step counters start over.')
                          : (language === 'zh-CN' ? '不从已有训练状态恢复。' : 'Does not restore an earlier training state.')}
                    />
                    {resumeMode !== 'none' ? (
                      <Form.Item
                        name="resume_checkpoint"
                        label={language === 'zh-CN' ? '恢复 Checkpoint' : 'Resume checkpoint'}
                        rules={[{ required: true }]}
                      >
                        <AutoComplete
                          options={(checkpointQuery.data ?? []).map((item) => ({ value: item.path, label: `${item.experiment_name ?? item.experiment_id ?? t('checkpoints.experiment')} · ${t('checkpoints.step')} ${item.step ?? '—'}` }))}
                          placeholder="/path/to/checkpoint"
                          filterOption={(input, option) => String(option?.value ?? '').toLowerCase().includes(input.toLowerCase())}
                        />
                      </Form.Item>
                    ) : null}
                    <div className="form-grid-3">
                      <Form.Item name="batch_size" label={t('builder.batchSize')} rules={[{ required: true }]}><InputNumber min={1} className="full-width" /></Form.Item>
                      <Form.Item name="learning_rate" label={t('builder.learningRate')} rules={[{ required: true }]}><InputNumber min={0} step={0.00001} className="full-width" /></Form.Item>
                      <Form.Item name="max_steps" label={t('builder.maxSteps')} rules={[{ required: true }]}><InputNumber min={1} className="full-width" /></Form.Item>
                      <Form.Item name="save_interval" label={t('builder.saveInterval')} rules={[{ required: true }]}><InputNumber min={1} className="full-width" /></Form.Item>
                      <Form.Item name="seed" label={t('builder.seed')}><InputNumber className="full-width" /></Form.Item>
                      <Form.Item name="num_workers" label={t('builder.workers')}><InputNumber min={0} className="full-width" /></Form.Item>
                    </div>
                    {params.length ? (
                      <Collapse ghost items={[{ key: 'module', label: t('builder.moduleParameters'), children: <div className="form-grid-2">{params.map((schema) => <DynamicField key={schema.key} schema={schema} language={language} />)}</div> }]} />
                    ) : null}
                  </Card>
                </Col>
                <Col xs={24} xl={8}>
                  <Card size="small" title={t('builder.resourceStrategy')}>
                    {singleGpu ? (
                      <Alert
                        className="single-gpu-notice"
                        type="success"
                        showIcon
                        message={t('builder.singleGpuDetected')}
                        description={t('builder.singleGpuDescription', {
                          index: selectableGpus[0]?.index ?? 0,
                          name: selectableGpus[0]?.name ?? 'GPU',
                        })}
                      />
                    ) : (
                      <>
                        <Form.Item name="resource_strategy">
                          <Radio.Group
                            optionType="button"
                            buttonStyle="solid"
                            options={[
                              { label: t('builder.autoGpu'), value: 'auto' },
                              { label: t('builder.fixedGpu'), value: 'fixed', disabled: detectedGpuCount === 0 },
                            ]}
                          />
                        </Form.Item>
                        <Form.Item name="gpu_count" label={t('builder.gpuCount')} rules={[{ required: true }]}>
                          <InputNumber min={1} max={detectedGpuCount && detectedGpuCount > 0 ? detectedGpuCount : 64} className="full-width" />
                        </Form.Item>
                        {resourceStrategy === 'fixed' ? (
                          <Form.Item
                            name="gpu_ids"
                            label={t('builder.gpuIds')}
                            rules={[
                              { required: true },
                              ({ getFieldValue }) => ({
                                validator: async (_, value: number[] | undefined) => {
                                  if ((value?.length ?? 0) === getFieldValue('gpu_count')) return;
                                  throw new Error(t('builder.gpuSelectionCount'));
                                },
                              }),
                            ]}
                          >
                            <Select
                              mode="multiple"
                              placeholder="0, 1"
                              onChange={(values: number[]) => form.setFieldValue('gpu_count', values.length)}
                              options={selectableGpus.map((gpu) => ({ value: gpu.index, label: `GPU ${gpu.index} · ${gpu.name}${gpu.available ? '' : ` · ${t('dashboard.occupied')}`}` }))}
                            />
                          </Form.Item>
                        ) : null}
                        {detectedGpuCount === 0 ? <Alert className="no-gpu-notice" type="warning" showIcon message={t('builder.noGpuDetected')} /> : null}
                      </>
                    )}
                    <Alert type="info" showIcon message="FIFO" description={t('builder.fifoHint')} />
                  </Card>
                </Col>
              </Row>
            </div>

            <div hidden={step !== 2} className="builder-step expert-step">
              <Alert
                type="info"
                showIcon
                message={t('builder.expertGuideTitle')}
                description={t('builder.expertGuideIntro')}
              />
              <Row gutter={[14, 14]} className="expert-guide-grid">
                <Col xs={24} lg={8}>
                  <Card size="small" title={t('builder.expertUseTitle')}>
                    <Typography.Paragraph>{t('builder.expertUseDescription')}</Typography.Paragraph>
                  </Card>
                </Col>
                <Col xs={24} lg={8}>
                  <Card size="small" title={t('builder.expertPriorityTitle')}>
                    <Typography.Paragraph>{t('builder.expertPriorityDescription')}</Typography.Paragraph>
                  </Card>
                </Col>
                <Col xs={24} lg={8}>
                  <Card size="small" title={t('builder.expertSafetyTitle')}>
                    <Typography.Paragraph>{t('builder.expertSafetyDescription')}</Typography.Paragraph>
                  </Card>
                </Col>
              </Row>
              <Collapse
                className="expert-example"
                size="small"
                items={[{
                  key: 'example',
                  label: t('builder.expertExampleTitle'),
                  children: (
                    <>
                      <Typography.Paragraph type="secondary">{t('builder.expertExampleDescription')}</Typography.Paragraph>
                      <pre>{t('builder.expertExampleYaml')}</pre>
                    </>
                  ),
                }]}
              />
              <Typography.Paragraph type="secondary" className="expert-editor-hint">{t('builder.yamlHint')}</Typography.Paragraph>
              <div className="editor-frame">
                <Editor
                  height="420px"
                  language="yaml"
                  theme={theme === 'dark' ? 'vs-dark' : 'light'}
                  value={expertYaml}
                  onChange={(value) => { setExpertYaml(value ?? ''); setResolved(undefined); setPreflight(undefined); }}
                  options={{ minimap: { enabled: false }, fontSize: 14, tabSize: 2, automaticLayout: true, scrollBeyondLastLine: false }}
                />
              </div>
            </div>

            <div hidden={step !== 3} className="builder-step review-step">
              <Row gutter={[18, 18]}>
                <Col xs={24} xl={10}>
                  <Card size="small" title={t('builder.review')}>
                    <Descriptions column={1} size="small" bordered>
                      <Descriptions.Item label={t('builder.experimentName')}>{Form.useWatch('name', form) || '—'}</Descriptions.Item>
                      <Descriptions.Item label={t('builder.backbone')}>{named(all.find((item) => item.id === backboneId) ?? { name: '—' } as Capability, language)}</Descriptions.Item>
                      <Descriptions.Item label={t('builder.actionHead')}>{named(all.find((item) => item.id === actionHeadId) ?? { name: '—' } as Capability, language)}</Descriptions.Item>
                      <Descriptions.Item label={t('builder.method')}>{named(all.find((item) => item.id === methodId) ?? { name: '—' } as Capability, language)}</Descriptions.Item>
                      <Descriptions.Item label={t('builder.dataset')}>{named(all.find((item) => item.id === selectedIds[3]) ?? { name: '—' } as Capability, language)}</Descriptions.Item>
                      {selectedWorkflow ? (
                        <Descriptions.Item label="Workflow">
                          {selectedWorkflow.title?.[language] ?? selectedWorkflow.title?.['en-US'] ?? selectedWorkflow.id}
                        </Descriptions.Item>
                      ) : null}
                      {datasetPath && datasetSource === 'local' ? <Descriptions.Item label={t('builder.datasetPath')}>{datasetPath}</Descriptions.Item> : null}
                      {datasetRegistrationId ? <Descriptions.Item label={t('builder.registeredDataset')}>{datasetRegistrationId}</Descriptions.Item> : null}
                      {datasetMixtureId ? <Descriptions.Item label={t('builder.datasetMixture')}>{datasetMixtureId}</Descriptions.Item> : null}
                      <Descriptions.Item label={language === 'zh-CN' ? '恢复方式' : 'Resume mode'}>
                        {resumeMode === 'full_state'
                          ? (language === 'zh-CN' ? '恢复完整训练状态' : 'Full training state')
                          : resumeMode === 'weights_only'
                            ? (language === 'zh-CN' ? '仅加载权重' : 'Weights only')
                            : (language === 'zh-CN' ? '从头训练' : 'Fresh run')}
                      </Descriptions.Item>
                      <Descriptions.Item label={t('builder.resourceStrategy')}>
                        {resourceStrategy === 'fixed' ? t('builder.fixedGpu') : t('builder.autoGpu')} · {Form.useWatch('gpu_count', form)} GPU
                      </Descriptions.Item>
                      <Descriptions.Item label="W&B">
                        {wandbEnabled ? (
                          <Space size={[4, 4]} wrap>
                            <Tag color={wandbMode === 'offline' ? 'blue' : 'green'}>{t(wandbMode === 'offline' ? 'builder.wandbOffline' : 'builder.wandbOnline')}</Tag>
                            {(selectedWandbCategories ?? []).map((id) => {
                              const category = wandbCategories.find((item) => item.id === id);
                              return <Tag key={id}>{localizedWandbText(category?.title, language, id)}</Tag>;
                            })}
                          </Space>
                        ) : t('common.disabled')}
                      </Descriptions.Item>
                    </Descriptions>
                    {hasExperimental ? <Alert className="step-alert" type="warning" showIcon message={t('builder.experimental')} description={t('builder.experimentalRisk')} /> : null}
                    <Space wrap className="review-actions">
                      <Button icon={<CodeOutlined />} onClick={() => resolveMutation.mutate()} loading={resolveMutation.isPending}>{t('builder.resolve')}</Button>
                      <Button type="primary" ghost icon={<CheckCircleFilled />} onClick={() => preflightMutation.mutate()} loading={preflightMutation.isPending}>{t('builder.preflight')}</Button>
                    </Space>
                  </Card>
                  {preflight ? (
                    <Card size="small" className="preflight-card">
                      <Result
                        status={preflight.ok ? 'success' : 'error'}
                        title={preflight.ok ? t('builder.preflightPassed') : t('builder.preflightFailed')}
                      />
                      <List
                        dataSource={preflight.items}
                        renderItem={(item) => (
                          <List.Item>
                            <List.Item.Meta
                              avatar={item.level === 'error' ? <WarningFilled className="danger" /> : <CheckCircleFilled className={item.level === 'warning' ? 'warning' : 'success'} />}
                              title={issueMessage(item)}
                              description={item.path}
                            />
                          </List.Item>
                        )}
                      />
                    </Card>
                  ) : null}
                </Col>
                <Col xs={24} xl={14}>
                  <Card size="small" title={t('builder.resolvedConfig')}>
                    {resolved ? (
                      <>
                        {resolved.command ? <div><Typography.Text type="secondary">{t('builder.command')}</Typography.Text><pre className="command-preview">{resolved.command}</pre></div> : null}
                        <pre className="config-preview">{YAML.stringify(resolved.resolved_config)}</pre>
                      </>
                    ) : <div className="empty-resolve"><CodeOutlined /><Typography.Text type="secondary">{t('builder.resolve')}</Typography.Text></div>}
                  </Card>
                </Col>
              </Row>
            </div>
          </Form>
        </AsyncState>
        <Divider />
        <div className="builder-footer">
          <Button icon={<ArrowLeftOutlined />} disabled={step === 0} onClick={() => setStep((value) => value - 1)}>{t('common.previous')}</Button>
          <Space>
            {step < 3 ? <Button type="primary" disabled={!stepComplete[step]} onClick={() => void validateAndAdvance()}>{t('common.next')} <ArrowRightOutlined /></Button> : (
              <Button
                type="primary"
                size="large"
                icon={<RocketOutlined />}
                disabled={!preflight?.ok}
                loading={submitMutation.isPending}
                onClick={submit}
              >
                {t('builder.submit')}
              </Button>
            )}
            {step === 3 ? <Button icon={<SaveOutlined />} loading={saveTemplateMutation.isPending} onClick={() => void saveTemplate()}>{t('templates.create')}</Button> : null}
          </Space>
        </div>
      </Card>
      <TemplateDuplicateModal
        detail={pendingTemplateDuplicate?.detail}
        saving={saveTemplateMutation.isPending}
        onCancel={() => setPendingTemplateDuplicate(undefined)}
        onOpenExisting={openDuplicateTemplate}
        onSaveAnyway={() => pendingTemplateDuplicate && saveTemplateMutation.mutate({ spec: pendingTemplateDuplicate.spec, allowDuplicate: true })}
      />
    </div>
  );
}
