import {
  BarChartOutlined,
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
  EvaluationBenchmark,
  EvaluationCheckpointSource,
  EvaluationCreateRequest,
  EvaluationCreateResult,
  EvaluationGroup,
  EvaluationKind,
  EvaluationPreset,
  EvaluationRun,
  EvaluationWandbConfig,
  ParameterSchema,
  PreflightItem,
  PreflightResult,
} from '../api/types';
import { usePreferences } from '../app-context';
import { useGpuRefreshInterval } from '../hooks/useGpuRefreshInterval';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { ServerDirectoryPicker } from '../components/ServerDirectoryPicker';
import { StatusTag } from '../components/StatusTag';
import {
  comparisonIssue,
  normalizeEvaluationResources,
  parameterHasValue,
  presetParameters,
} from './evaluation-form';

interface EvaluationFormValues {
  name: string;
  kind: EvaluationKind;
  source_kind: 'temporary_checkpoint' | 'managed_deployment';
  deployment_id?: string;
  checkpoint_kind: 'indexed' | 'local';
  checkpoint_id?: string;
  batch_checkpoint_ids?: string[];
  checkpoint_path?: string;
  combination_id?: string;
  benchmark_id?: string;
  preset: EvaluationPreset;
  suite?: string;
  suites?: string[];
  task_set?: string;
  split?: string;
  parameters: Record<string, unknown>;
  resources: EvaluationCreateRequest['resources'];
  wandb: EvaluationWandbConfig;
  acknowledge_experimental: boolean;
}

function localized(language: string, english?: string, chinese?: string, fallback = ''): string {
  return language === 'zh-CN' ? chinese || english || fallback : english || chinese || fallback;
}

function successRate(run: EvaluationRun): string {
  const value = run.result?.summary.success_rate;
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function rateFromSummary(summary: Record<string, unknown>): string {
  const value = summary.success_rate;
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}

function DynamicEvaluationParameter({ schema }: { schema: ParameterSchema }) {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const label = localized(language, schema.label, schema.label_zh, schema.key);
  const description = localized(language, schema.description, schema.description_zh);
  let control;
  if (schema.type === 'boolean') control = <Switch />;
  else if (schema.type === 'number' || schema.type === 'integer') {
    control = <InputNumber className="full-width" min={schema.min} max={schema.max} precision={schema.type === 'integer' ? 0 : undefined} />;
  } else if (schema.type === 'select') {
    control = <Select options={(schema.options ?? []).map((option) => ({ value: option.value, label: localized(language, option.label, option.label_zh, String(option.value)) }))} />;
  } else if (schema.type === 'textarea') control = <Input.TextArea autoSize={{ minRows: 3, maxRows: 8 }} />;
  else if (schema.type === 'path') control = <ServerDirectoryPicker placeholder="/path/on/server" />;
  else control = <Input />;
  return (
    <Form.Item
      name={['parameters', schema.key]}
      label={label}
      extra={description || undefined}
      valuePropName={schema.type === 'boolean' ? 'checked' : 'value'}
      rules={schema.required && schema.type !== 'boolean' ? [{ required: true, message: t('evaluation.parameterRequired', { name: label }) }] : undefined}
    >
      {control}
    </Form.Item>
  );
}

function benchmarkOptionLabel(benchmark: EvaluationBenchmark, language: string, verified: string, experimental: string, unavailable: string) {
  return (
    <Space size={6}>
      <span>{localized(language, benchmark.name, benchmark.name_zh, benchmark.id)}</span>
      <Tag color={benchmark.status === 'experimental' ? 'orange' : 'green'}>{benchmark.status === 'experimental' ? experimental : verified}</Tag>
      {!benchmark.available ? <Tag color="red">{unavailable}</Tag> : null}
    </Space>
  );
}

function inspectionIssueDescription(item: PreflightItem): string | undefined {
  const detail = item.detail ?? {};
  const values = ['environment_key', 'configured_path', 'resolved_path']
    .map((key) => detail[key])
    .filter((value): value is string => typeof value === 'string' && Boolean(value));
  return values.length ? values.join(' · ') : undefined;
}

function CreateEvaluationModal({ open, onClose, onCreated }: {
  open: boolean;
  onClose: () => void;
  onCreated: (result: EvaluationCreateResult) => void;
}) {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const gpuRefreshInterval = useGpuRefreshInterval();
  const [form] = Form.useForm<EvaluationFormValues>();
  const [step, setStep] = useState(0);
  const [preflightResult, setPreflightResult] = useState<PreflightResult>();
  const [searchParams] = useSearchParams();
  const benchmarks = useQuery({ queryKey: ['evaluation-benchmarks'], queryFn: api.evaluations.benchmarks, enabled: open });
  const deploymentCapabilities = useQuery({ queryKey: ['deployment-capabilities'], queryFn: api.deployments.capabilities, enabled: open });
  const checkpoints = useQuery({ queryKey: ['checkpoints'], queryFn: api.checkpoints, enabled: open });
  const deployments = useQuery({ queryKey: ['deployments'], queryFn: () => api.deployments.list(), enabled: open });
  const gpus = useQuery({ queryKey: ['gpus'], queryFn: api.gpus, enabled: open, refetchInterval: open ? gpuRefreshInterval : false });
  const values = Form.useWatch([], form) as EvaluationFormValues | undefined;
  const checkpointKind = values?.checkpoint_kind ?? 'indexed';
  const evaluationKind = values?.kind ?? 'standard';
  const sourceKind = values?.source_kind ?? 'temporary_checkpoint';
  const managedSource = sourceKind === 'managed_deployment';
  const checkpointId = evaluationKind === 'batch' ? values?.batch_checkpoint_ids?.[0] : values?.checkpoint_id;
  const localCheckpointPath = values?.checkpoint_path?.trim() ?? '';
  const [debouncedLocalCheckpointPath, setDebouncedLocalCheckpointPath] = useState('');
  const visibleGpus = healthyGpus(gpus.data);
  const visibleGpuCount = visibleGpus.length;
  const singleGpu = visibleGpuCount === 1;
  const selectedDeployment = deployments.data?.find((item) => item.id === values?.deployment_id);
  const effectiveCombinationId = managedSource ? selectedDeployment?.combination_id : values?.combination_id;
  const selectedCombination = deploymentCapabilities.data?.combinations.find((item) => item.id === effectiveCombinationId);
  const selectedBenchmark = benchmarks.data?.benchmarks.find((item) => item.id === values?.benchmark_id);
  const preset = values?.preset ?? 'quick';

  useEffect(() => {
    if (checkpointKind !== 'local' || !localCheckpointPath) {
      setDebouncedLocalCheckpointPath('');
      return;
    }
    const timer = window.setTimeout(() => setDebouncedLocalCheckpointPath(localCheckpointPath), 500);
    return () => window.clearTimeout(timer);
  }, [checkpointKind, localCheckpointPath]);

  const inspectionSource = useMemo<EvaluationCheckpointSource | undefined>(() => {
    if (checkpointKind === 'indexed') {
      return checkpointId ? { kind: 'indexed', checkpoint_id: checkpointId } : undefined;
    }
    return debouncedLocalCheckpointPath
      ? { kind: 'local', path: debouncedLocalCheckpointPath }
      : undefined;
  }, [checkpointId, checkpointKind, debouncedLocalCheckpointPath]);
  const inspectionMatchesCurrentSource = Boolean(inspectionSource
    && (inspectionSource.kind === 'indexed' || inspectionSource.path === localCheckpointPath));
  const checkpointInspection = useQuery({
    queryKey: ['evaluation-checkpoint-inspection', inspectionSource, values?.combination_id ?? 'auto'],
    queryFn: () => api.evaluations.inspectCheckpoint(inspectionSource!, values?.combination_id),
    enabled: open && !managedSource && Boolean(inspectionSource) && inspectionMatchesCurrentSource,
    retry: false,
  });
  const activeInspection = inspectionMatchesCurrentSource ? checkpointInspection.data : undefined;

  const connectedCombinations = useMemo(() => {
    const allBenchmarks = benchmarks.data?.benchmarks ?? [];
    return (deploymentCapabilities.data?.combinations ?? []).filter((combination) =>
      allBenchmarks.some((benchmark) => !benchmark.compatibility.combination_ids.length || benchmark.compatibility.combination_ids.includes(combination.id)),
    );
  }, [benchmarks.data?.benchmarks, deploymentCapabilities.data?.combinations]);
  const candidateCombinationIds = useMemo(() => Array.from(new Set(
    (activeInspection?.evaluation_candidates ?? []).map((candidate) => candidate.combination_id),
  )), [activeInspection?.evaluation_candidates]);
  const candidateCombinations = useMemo(() => {
    const allowed = new Set(candidateCombinationIds);
    return connectedCombinations.filter((combination) => allowed.has(combination.id));
  }, [candidateCombinationIds, connectedCombinations]);
  const selectedInspectionCandidate = activeInspection?.evaluation_candidates.find(
    (candidate) => candidate.combination_id === values?.combination_id,
  );
  const inspectedBenchmarkIds = selectedInspectionCandidate?.benchmark_ids
    ?? (candidateCombinationIds.length === 1 ? activeInspection?.compatible_benchmark_ids : undefined)
    ?? [];
  const compatibleBenchmarks = useMemo(() => (benchmarks.data?.benchmarks ?? []).filter((benchmark) =>
    Boolean(effectiveCombinationId)
    && (managedSource || inspectedBenchmarkIds.includes(benchmark.id))
    && (!benchmark.compatibility.combination_ids.length || benchmark.compatibility.combination_ids.includes(effectiveCombinationId!))),
  [benchmarks.data?.benchmarks, effectiveCombinationId, inspectedBenchmarkIds, managedSource]);

  useEffect(() => {
    if (!open) return;
    const checkpointId = searchParams.get('checkpoint_id') ?? undefined;
    const checkpointPath = searchParams.get('checkpoint') ?? undefined;
    form.resetFields();
    form.setFieldsValue({
      name: '',
      kind: 'standard',
      source_kind: 'temporary_checkpoint',
      checkpoint_kind: checkpointId ? 'indexed' : checkpointPath ? 'local' : 'indexed',
      checkpoint_id: checkpointId,
      checkpoint_path: checkpointPath,
      preset: 'quick',
      parameters: {},
      resources: { strategy: 'auto', gpu_count: 1, gpu_ids: [] },
      wandb: { enabled: false, mode: 'online', project: 'AlphaBrain-Evaluation', entity: '', run_name: '', tags: [], categories: ['summary', 'tasks', 'config'] },
      acknowledge_experimental: false,
    });
    setStep(0);
    setPreflightResult(undefined);
  }, [form, open, searchParams]);

  useEffect(() => {
    if (managedSource && !['standard', 'batch', 'world_model_video'].includes(evaluationKind)) {
      form.setFieldValue('source_kind', 'temporary_checkpoint');
    }
  }, [evaluationKind, form, managedSource]);

  useEffect(() => {
    if (!managedSource || !selectedDeployment) return;
    form.setFieldsValue({ combination_id: selectedDeployment.combination_id, benchmark_id: undefined });
    setPreflightResult(undefined);
  }, [form, managedSource, selectedDeployment]);

  useEffect(() => {
    if (!open || managedSource) return;
    form.setFieldsValue({ combination_id: undefined, benchmark_id: undefined });
    setPreflightResult(undefined);
  }, [checkpointId, checkpointKind, form, localCheckpointPath, managedSource, open]);

  useEffect(() => {
    if (!activeInspection || checkpointInspection.isFetching) return;
    const ids = candidateCombinations.map((combination) => combination.id);
    const current = form.getFieldValue('combination_id');
    if (ids.length === 1 && current !== ids[0]) {
      form.setFieldValue('combination_id', ids[0]);
    } else if (current && !ids.includes(current)) {
      form.setFieldValue('combination_id', undefined);
    }
  }, [activeInspection, candidateCombinations, checkpointInspection.isFetching, form]);

  useEffect(() => {
    const benchmarkId = form.getFieldValue('benchmark_id');
    if (benchmarkId && selectedInspectionCandidate && !selectedInspectionCandidate.benchmark_ids.includes(benchmarkId)) {
      form.setFieldValue('benchmark_id', undefined);
    }
  }, [form, selectedInspectionCandidate]);

  useEffect(() => {
    if (!selectedBenchmark) return;
    const defaults = selectedBenchmark.defaults;
    form.setFieldsValue({
      suite: String(defaults.suite ?? selectedBenchmark.suites[0]?.value ?? '') || undefined,
      suites: evaluationKind === 'batch'
        ? [String(defaults.suite ?? selectedBenchmark.suites[0]?.value ?? '')].filter(Boolean)
        : undefined,
      task_set: String(defaults.task_set ?? selectedBenchmark.task_sets[0]?.value ?? '') || undefined,
      split: String(defaults.split ?? selectedBenchmark.splits[0]?.value ?? '') || undefined,
      preset: 'quick',
      parameters: presetParameters(selectedBenchmark, 'quick'),
      acknowledge_experimental: false,
    });
    setPreflightResult(undefined);
  }, [evaluationKind, form, selectedBenchmark]);

  useEffect(() => {
    if (!selectedBenchmark) return;
    form.setFieldValue('parameters', presetParameters(selectedBenchmark, preset));
    setPreflightResult(undefined);
  }, [form, preset, selectedBenchmark]);

  useEffect(() => {
    if (!open || !singleGpu) return;
    form.setFieldValue('resources', normalizeEvaluationResources(form.getFieldValue('resources'), visibleGpuCount));
  }, [form, open, singleGpu, visibleGpuCount]);

  const makePayload = async (): Promise<EvaluationCreateRequest> => {
    const data = await form.validateFields();
    const parameters = { ...(data.parameters ?? {}) };
    delete parameters.suite;
    delete parameters.task_set;
    delete parameters.split;
    const indexedSources = data.kind === 'batch' && data.checkpoint_kind === 'indexed'
      ? (data.batch_checkpoint_ids ?? []).map((checkpoint_id) => ({ kind: 'indexed' as const, checkpoint_id }))
      : [];
    const checkpointSource = data.source_kind === 'managed_deployment'
      ? undefined
      : data.checkpoint_kind === 'indexed'
        ? indexedSources[0] ?? { kind: 'indexed' as const, checkpoint_id: data.checkpoint_id! }
        : { kind: 'local' as const, path: data.checkpoint_path!.trim() };
    return {
      name: data.name.trim(),
      kind: data.kind,
      source_kind: data.source_kind,
      ...(checkpointSource ? { checkpoint_source: checkpointSource } : {}),
      ...(indexedSources.length ? { checkpoint_sources: indexedSources } : {}),
      ...(data.suites?.length ? { suites: data.suites } : {}),
      ...(data.deployment_id ? { deployment_id: data.deployment_id } : {}),
      ...(effectiveCombinationId ? { combination_id: effectiveCombinationId } : {}),
      benchmark_id: data.benchmark_id!,
      preset: data.preset,
      ...(data.suite ? { suite: data.suite } : {}),
      ...(data.task_set ? { task_set: data.task_set } : {}),
      ...(data.split ? { split: data.split } : {}),
      parameters,
      resources: normalizeEvaluationResources(data.resources, visibleGpuCount, ['cl_matrix', 'rl_iterations'].includes(data.kind)),
      wandb: { ...data.wandb, enabled: Boolean(data.wandb?.enabled), categories: data.wandb?.categories ?? [] },
      acknowledge_experimental: Boolean(data.acknowledge_experimental),
    };
  };
  const preflight = useMutation({
    mutationFn: async () => api.evaluations.preflight(await makePayload()),
    onSuccess: setPreflightResult,
    onError: (error) => message.error(error.message),
  });
  const create = useMutation({
    mutationFn: async () => api.evaluations.create(await makePayload()),
    onSuccess: (result) => { message.success(t('evaluation.created')); onCreated(result); },
    onError: (error) => message.error(error.message),
  });

  const checkpointSourcePresent = managedSource
    ? Boolean(selectedDeployment?.status === 'running' && selectedDeployment.managed_inference_available)
    : Boolean(checkpointKind === 'indexed'
      ? evaluationKind === 'batch' ? values?.batch_checkpoint_ids?.length : checkpointId
      : localCheckpointPath);
  const inspectionLoading = !managedSource && checkpointSourcePresent && (!inspectionMatchesCurrentSource || checkpointInspection.isFetching);
  const visibleInspectionIssues = (activeInspection?.issues ?? []).filter((item) =>
    !(item.id === 'deployment_combination_ambiguous' && candidateCombinationIds.length > 1),
  );
  const blockingInspectionIssues = visibleInspectionIssues.filter((item) => item.level === 'error');
  const selectedCandidateAvailable = Boolean(values?.combination_id
    && candidateCombinations.some((combination) => combination.id === values.combination_id));
  const inspectionReady = managedSource ? checkpointSourcePresent : Boolean(checkpointSourcePresent
    && inspectionMatchesCurrentSource
    && !checkpointInspection.isFetching
    && !checkpointInspection.error
    && activeInspection
    && candidateCombinations.length
    && selectedCandidateAvailable
    && !blockingInspectionIssues.length);
  const sourceReady = Boolean(values?.name?.trim() && inspectionReady);
  const requiresAcknowledgement = selectedCombination?.status === 'experimental' || selectedBenchmark?.status === 'experimental';
  const specializedReady = evaluationKind === 'cl_matrix'
    ? Boolean(String(values?.parameters?.run_id ?? '').trim())
    : evaluationKind === 'rl_iterations'
      ? Boolean(String(values?.parameters?.run_dir ?? '').trim())
      : true;
  const benchmarkReady = Boolean(selectedBenchmark
    && selectedBenchmark.available
    && values?.preset
    && (!selectedBenchmark.suites.length || (evaluationKind === 'batch' ? values?.suites?.length : values?.suite))
    && (!selectedBenchmark.task_sets.length || values?.task_set)
    && (!selectedBenchmark.splits.length || values?.split)
    && (preset !== 'custom' || selectedBenchmark.parameters.every((schema) => parameterHasValue(values?.parameters, schema)))
    && specializedReady
    && (!requiresAcknowledgement || values?.acknowledge_experimental));
  const requestedGpuCount = ['cl_matrix', 'rl_iterations'].includes(evaluationKind) ? Number(values?.resources?.gpu_count ?? 1) : 1;
  const resourceReady = Boolean(singleGpu || values?.resources?.strategy === 'auto' || values?.resources?.gpu_ids?.length === requestedGpuCount);
  const wandbReady = Boolean(!values?.wandb?.enabled || (values.wandb.project?.trim() && values.wandb.categories?.length));
  const nextDisabled = step === 0 ? !sourceReady : step === 1 ? !benchmarkReady : step === 2 ? !(resourceReady && wandbReady) : true;

  const next = async () => {
    if (step === 0 && !inspectionReady) return;
    const fields: Parameters<typeof form.validateFields>[0] = step === 0
      ? [['name'], ['kind'], ['source_kind'], ['deployment_id'], ['checkpoint_id'], ['batch_checkpoint_ids'], ['checkpoint_path'], ['combination_id']]
      : step === 1 ? [['benchmark_id'], ['preset'], ['suite'], ['task_set'], ['split'], ['parameters'], ['acknowledge_experimental']]
        : [['resources'], ['wandb']];
    try {
      await form.validateFields(fields);
      setStep((current) => Math.min(3, current + 1));
    } catch { /* Form renders the errors. */ }
  };

  return (
    <Modal
      className="evaluation-create-modal"
      open={open}
      width={980}
      title={t('evaluation.createTitle')}
      onCancel={onClose}
      footer={<div className="builder-footer"><Button disabled={step === 0} onClick={() => setStep((value) => Math.max(0, value - 1))}>{t('common.previous')}</Button><Space><Button onClick={onClose}>{t('common.cancel')}</Button>{step < 3 ? <Button type="primary" disabled={nextDisabled} onClick={() => void next()}>{t('common.next')}</Button> : null}{step === 3 ? <Button icon={<SafetyCertificateOutlined />} loading={preflight.isPending} onClick={() => preflight.mutate()}>{t('evaluation.runPreflight')}</Button> : null}{step === 3 ? <Button type="primary" icon={<ExperimentOutlined />} disabled={!preflightResult?.ok} loading={create.isPending} onClick={() => create.mutate()}>{t('evaluation.submit')}</Button> : null}</Space></div>}
    >
      <Steps current={step} onChange={(nextStep) => { if (nextStep < step) setStep(nextStep); }} items={[
        { title: t('evaluation.sourceStep') },
        { title: t('evaluation.benchmarkStep') },
        { title: t('evaluation.runtimeStep') },
        { title: t('evaluation.reviewStep') },
      ]} />
      <AsyncState
        loading={benchmarks.isLoading || deploymentCapabilities.isLoading || checkpoints.isLoading || deployments.isLoading || gpus.isLoading}
        error={benchmarks.error ?? deploymentCapabilities.error ?? checkpoints.error ?? deployments.error ?? gpus.error}
        onRetry={() => { void benchmarks.refetch(); void deploymentCapabilities.refetch(); void checkpoints.refetch(); void deployments.refetch(); void gpus.refetch(); }}
      >
        <Form<EvaluationFormValues> className="evaluation-form" form={form} layout="vertical" onValuesChange={() => setPreflightResult(undefined)}>
          {step === 0 ? <div className="evaluation-step">
            <Form.Item name="kind" label={t('evaluation.kindLabel')} rules={[{ required: true }]}><Select options={( ['standard', 'batch', 'cl_matrix', 'rl_iterations', 'online_stdp', 'world_model_video'] as EvaluationKind[]).map((kind) => ({ value: kind, label: t(`evaluation.kind.${kind}`) }))} /></Form.Item>
            <Form.Item name="name" label={t('evaluation.name')} rules={[{ required: true, whitespace: true }]}><Input placeholder={t('evaluation.namePlaceholder')} /></Form.Item>
            <Form.Item name="source_kind" label={t('evaluation.modelSource')}><Radio.Group optionType="button" buttonStyle="solid" options={[{ label: t('evaluation.temporaryCheckpoint'), value: 'temporary_checkpoint' }, { label: t('evaluation.managedDeployment'), value: 'managed_deployment', disabled: !['standard', 'batch', 'world_model_video'].includes(evaluationKind) }]} /></Form.Item>
            {managedSource ? <>
              <Form.Item name="deployment_id" label={t('evaluation.managedDeployment')} rules={[{ required: true }]}><Select showSearch optionFilterProp="searchText" placeholder={t('evaluation.selectManagedDeployment')} options={(deployments.data ?? []).filter((deployment) => deployment.status === 'running' && deployment.managed_inference_available).map((deployment) => ({ value: deployment.id, label: `${deployment.name} · ${deployment.combination_id} · ${deployment.endpoint.port ?? '—'}`, searchText: `${deployment.name} ${deployment.combination_id} ${deployment.checkpoint_path ?? ''}` }))} /></Form.Item>
              {selectedDeployment ? <Alert type="success" showIcon message={t('evaluation.reusingDeployment')} description={`${selectedDeployment.combination_id} · GPU ${selectedDeployment.assigned_gpu_ids.join(', ')} · ${selectedDeployment.endpoint.port ?? '—'}`} /> : <Alert type="info" showIcon message={t('evaluation.managedDeploymentHint')} />}
            </> : <>
              <Form.Item name="checkpoint_kind" label={t('evaluation.checkpointSource')}><Radio.Group optionType="button" buttonStyle="solid" options={[{ label: t('evaluation.indexedCheckpoint'), value: 'indexed' }, { label: t('evaluation.localCheckpoint'), value: 'local' }]} /></Form.Item>
              {checkpointKind === 'indexed' ? evaluationKind === 'batch'
                ? <Form.Item name="batch_checkpoint_ids" label={t('evaluation.checkpoints')} rules={[{ required: true }]}><Select mode="multiple" showSearch optionFilterProp="searchText" placeholder={t('evaluation.selectCheckpoints')} options={(checkpoints.data ?? []).filter((checkpoint) => checkpoint.complete !== false).map((checkpoint) => ({ value: checkpoint.id, label: checkpoint.experiment_name ? `${checkpoint.experiment_name} · ${checkpoint.path}` : checkpoint.path, searchText: `${checkpoint.experiment_name ?? ''} ${checkpoint.path}` }))} /></Form.Item>
                : <Form.Item name="checkpoint_id" label={t('evaluation.checkpoint')} rules={[{ required: true }]}><Select showSearch optionFilterProp="searchText" placeholder={t('evaluation.selectCheckpoint')} options={(checkpoints.data ?? []).filter((checkpoint) => checkpoint.complete !== false).map((checkpoint) => ({ value: checkpoint.id, label: checkpoint.experiment_name ? `${checkpoint.experiment_name} · ${checkpoint.path}` : checkpoint.path, searchText: `${checkpoint.experiment_name ?? ''} ${checkpoint.path}` }))} /></Form.Item>
                : <Form.Item name="checkpoint_path" label={t('evaluation.checkpointPath')} rules={[{ required: true, whitespace: true }]}><ServerDirectoryPicker placeholder="/path/to/checkpoint" /></Form.Item>}
              {inspectionLoading ? <Alert type="info" showIcon message={t('evaluation.checkpointInspecting')} /> : null}
              {checkpointInspection.error && inspectionMatchesCurrentSource ? <Alert type="error" showIcon message={t('evaluation.checkpointInspectionFailed')} description={checkpointInspection.error.message} /> : null}
              {activeInspection && !checkpointInspection.isFetching && candidateCombinations.length === 0 ? <Alert type="error" showIcon message={t('evaluation.checkpointNoCandidates')} /> : null}
              {activeInspection && !checkpointInspection.isFetching && candidateCombinations.length === 1 && !blockingInspectionIssues.length ? <Alert type="success" showIcon message={t('evaluation.checkpointDetected')} description={t('evaluation.checkpointDetectedDescription', { combination: candidateCombinations[0].id })} /> : null}
              {activeInspection && !checkpointInspection.isFetching && candidateCombinations.length > 1 && !values?.combination_id ? <Alert type="warning" showIcon message={t('evaluation.checkpointMultipleCandidates')} description={t('evaluation.checkpointMultipleCandidatesDescription', { count: candidateCombinations.length })} /> : null}
              {activeInspection && !checkpointInspection.isFetching ? visibleInspectionIssues.map((item, index) => <Alert key={`${item.id ?? item.title}-${index}`} type={item.level === 'error' ? 'error' : item.level === 'warning' ? 'warning' : item.level === 'success' ? 'success' : 'info'} showIcon message={item.message_i18n?.[language] ?? item.message ?? item.title} description={inspectionIssueDescription(item)} />) : null}
              <Form.Item name="combination_id" label={t('evaluation.modelCombination')} rules={[{ required: true }]}><Select disabled={!activeInspection || checkpointInspection.isFetching || candidateCombinations.length === 0} loading={inspectionLoading} showSearch optionFilterProp="searchText" placeholder={t('evaluation.selectCombination')} options={candidateCombinations.map((combination) => ({ value: combination.id, label: localized(language, combination.name, combination.name_zh, combination.id), searchText: `${combination.id} ${combination.name} ${combination.name_zh ?? ''} ${combination.backbone_id} ${combination.action_head_id}` }))} /></Form.Item>
              <Alert type="info" showIcon message={t('evaluation.temporaryServiceHint')} />
            </>}
          </div> : null}
          {step === 1 ? <div className="evaluation-step">
            <Form.Item name="benchmark_id" label={t('evaluation.benchmark')} rules={[{ required: true }]}><Select placeholder={t('evaluation.selectBenchmark')} options={compatibleBenchmarks.map((benchmark) => ({ value: benchmark.id, label: benchmarkOptionLabel(benchmark, language, t('capability.verified'), t('capability.experimental'), t('evaluation.unavailable')), disabled: !benchmark.available }))} /></Form.Item>
            {compatibleBenchmarks.some((benchmark) => !benchmark.available) ? <Alert type="warning" showIcon message={t('evaluation.environmentUnavailable')} description={compatibleBenchmarks.filter((benchmark) => !benchmark.available).map((benchmark) => `${localized(language, benchmark.name, benchmark.name_zh, benchmark.id)}: ${benchmark.environment.filter((item) => item.required && !item.configured).map((item) => item.id).join(', ') || t('evaluation.environmentCheckFailed')}`).join('\n')} action={<Button size="small" onClick={() => window.open('/settings?tab=environment', '_self')}>{t('settings.environment')}</Button>} /> : null}
            {selectedBenchmark ? <Card size="small" className="evaluation-benchmark-card"><Space wrap><Tag color={selectedBenchmark.status === 'experimental' ? 'orange' : 'green'}>{t(`capability.${selectedBenchmark.status}`)}</Tag><Typography.Text strong>{localized(language, selectedBenchmark.name, selectedBenchmark.name_zh, selectedBenchmark.id)}</Typography.Text></Space>{localized(language, selectedBenchmark.description, selectedBenchmark.description_zh) ? <Typography.Paragraph type="secondary">{localized(language, selectedBenchmark.description, selectedBenchmark.description_zh)}</Typography.Paragraph> : null}{selectedBenchmark.environment.filter((item) => item.required && !item.configured).map((item) => <Alert key={item.id} type="error" showIcon message={localized(language, item.title, item.title_zh, item.id)} description={localized(language, item.message, item.message_zh)} />)}</Card> : null}
            <Form.Item name="preset" label={t('evaluation.scale')} rules={[{ required: true }]}><Radio.Group className="evaluation-preset-grid">{(['quick', 'standard', 'full', 'custom'] as EvaluationPreset[]).map((id) => { const definition = selectedBenchmark?.presets.find((item) => item.id === id); return <Radio.Button key={id} value={id}><b>{definition ? localized(language, definition.name, definition.name_zh, t(`evaluation.preset.${id}`)) : t(`evaluation.preset.${id}`)}</b><small>{definition ? localized(language, definition.description, definition.description_zh, t(`evaluation.preset.${id}Desc`)) : t(`evaluation.preset.${id}Desc`)}</small></Radio.Button>; })}</Radio.Group></Form.Item>
            <div className="form-grid-3">
              {selectedBenchmark?.suites.length ? evaluationKind === 'batch'
                ? <Form.Item name="suites" label={t('evaluation.suites')} rules={[{ required: true }]}><Select mode="multiple" options={selectedBenchmark.suites.map((item) => ({ value: item.value, label: localized(language, item.label, item.label_zh, item.value) }))} /></Form.Item>
                : <Form.Item name="suite" label={t('evaluation.suite')} rules={[{ required: true }]}><Select options={selectedBenchmark.suites.map((item) => ({ value: item.value, label: localized(language, item.label, item.label_zh, item.value) }))} /></Form.Item> : null}
              {selectedBenchmark?.task_sets.length ? <Form.Item name="task_set" label={t('evaluation.taskSet')} rules={[{ required: true }]}><Select options={selectedBenchmark.task_sets.map((item) => ({ value: item.value, label: localized(language, item.label, item.label_zh, item.value) }))} /></Form.Item> : null}
              {selectedBenchmark?.splits.length ? <Form.Item name="split" label={t('evaluation.split')} rules={[{ required: true }]}><Select options={selectedBenchmark.splits.map((item) => ({ value: item.value, label: localized(language, item.label, item.label_zh, item.value) }))} /></Form.Item> : null}
            </div>
            {evaluationKind === 'batch' ? <Alert type="info" showIcon message={t('evaluation.batchHint')} description={t('evaluation.batchHintDescription', { checkpoints: managedSource ? 1 : values?.batch_checkpoint_ids?.length ?? 1, suites: values?.suites?.length ?? 0 })} /> : null}
            {evaluationKind === 'cl_matrix' ? <Card size="small" title={t('evaluation.specializedParameters')}><Alert type="info" showIcon message={t('evaluation.clMatrixHint')} /><div className="form-grid-2"><Form.Item name={['parameters', 'run_id']} label="CL run ID" rules={[{ required: true, whitespace: true }]}><Input placeholder="run_20260716" /></Form.Item><Form.Item name={['parameters', 'model']} label="Model"><Input /></Form.Item><Form.Item name={['parameters', 'trials']} label="Trials"><InputNumber className="full-width" min={1} max={1000} /></Form.Item><Form.Item name={['parameters', 'last_only']} label={t('evaluation.lastCheckpointOnly')} valuePropName="checked"><Switch /></Form.Item></div></Card> : null}
            {evaluationKind === 'rl_iterations' ? <Card size="small" title={t('evaluation.specializedParameters')}><Alert type="info" showIcon message={t('evaluation.rlIterationsHint')} /><div className="form-grid-2"><Form.Item name={['parameters', 'run_dir']} label={t('evaluation.rlRunDirectory')} rules={[{ required: true, whitespace: true }]}><ServerDirectoryPicker placeholder="results/training/rlt-run" /></Form.Item><Form.Item name={['parameters', 'task_ids']} label="Task IDs"><Input placeholder="0,1,2" /></Form.Item><Form.Item name={['parameters', 'n_eps']} label="Episodes"><InputNumber className="full-width" min={1} max={10000} /></Form.Item><Form.Item name={['parameters', 'num_workers']} label="Workers"><InputNumber className="full-width" min={1} max={128} /></Form.Item></div></Card> : null}
            {evaluationKind === 'online_stdp' ? <Card size="small" title={t('evaluation.specializedParameters')}><Alert type="warning" showIcon message={t('evaluation.onlineStdpHint')} /><div className="form-grid-2"><Form.Item name={['parameters', 'stdp_lr']} label="STDP learning rate"><InputNumber className="full-width" min={0} step={0.0001} /></Form.Item><Form.Item name={['parameters', 'stdp_warmup']} label="STDP warmup"><InputNumber className="full-width" min={0} precision={0} /></Form.Item><Form.Item name={['parameters', 'stdp_max_deviation']} label="Max weight deviation"><InputNumber className="full-width" min={0} /></Form.Item><Form.Item name={['parameters', 'stdp_rollback_shrink']} label="Rollback shrink"><InputNumber className="full-width" min={0} max={1} /></Form.Item></div></Card> : null}
            {evaluationKind === 'world_model_video' ? <Alert type="info" showIcon message={t('evaluation.worldModelVideoHint')} /> : null}
            {preset === 'custom' ? <><Alert type="warning" showIcon message={t('evaluation.customHint')} /><div className="form-grid-2">{selectedBenchmark?.parameters.filter((schema) => !['suite', 'task_set', 'split'].includes(schema.key) && !(evaluationKind === 'rl_iterations' && schema.key === 'task_ids')).map((schema) => <DynamicEvaluationParameter key={schema.key} schema={schema} />)}</div></> : null}
            {requiresAcknowledgement ? <><Alert type="warning" showIcon message={t('builder.experimental')} description={t('evaluation.experimentalRisk')} /><Form.Item name="acknowledge_experimental" valuePropName="checked" rules={[{ validator: (_, checked) => checked ? Promise.resolve() : Promise.reject(new Error(t('evaluation.experimentalConfirmationRequired'))) }]}><Checkbox>{t('evaluation.experimentalConfirmation')}</Checkbox></Form.Item></> : null}
          </div> : null}
          {step === 2 ? <div className="evaluation-step">
            <Typography.Title level={5}>{t('evaluation.gpuResources')}</Typography.Title>
            {managedSource ? <Alert type="info" showIcon message={t('evaluation.managedResourcesHint')} description={selectedDeployment ? `GPU ${selectedDeployment.assigned_gpu_ids.join(', ')}` : undefined} /> : singleGpu ? <Alert className="single-gpu-notice" type="info" showIcon message={t('builder.singleGpuDetected')} description={t('builder.singleGpuDescription', { index: visibleGpus[0]?.index, name: visibleGpus[0]?.name })} /> : <><Form.Item name={['resources', 'strategy']} label={t('builder.resourceStrategy')}><Radio.Group optionType="button" options={[{ label: t('builder.autoGpu'), value: 'auto' }, { label: t('builder.fixedGpu'), value: 'fixed' }]} /></Form.Item>{['cl_matrix', 'rl_iterations'].includes(evaluationKind) ? <Form.Item name={['resources', 'gpu_count']} label={t('builder.gpuCount')} rules={[{ required: true }]}><InputNumber className="full-width" min={1} max={Math.max(1, visibleGpuCount || 1)} precision={0} /></Form.Item> : null}{values?.resources?.strategy === 'fixed' ? <Form.Item name={['resources', 'gpu_ids']} label={t('builder.gpuIds')} rules={[{ required: true }]}><Select mode="multiple" maxCount={requestedGpuCount} options={visibleGpus.map((gpu) => ({ value: gpu.index, label: `GPU ${gpu.index} · ${gpu.name}` }))} /></Form.Item> : null}</>}
            {!managedSource ? <Alert type="info" showIcon message={t('evaluation.fifoHint')} /> : null}
            <Card size="small" className="evaluation-wandb-card" title={<Space><span>{t('builder.wandbTitle')}</span><Form.Item name={['wandb', 'enabled']} valuePropName="checked" noStyle><Switch /></Form.Item></Space>}>
              <Typography.Paragraph type="secondary">{t('evaluation.wandbDescription')}</Typography.Paragraph>
              {values?.wandb?.enabled ? <><div className="form-grid-2"><Form.Item name={['wandb', 'mode']} label={t('builder.wandbMode')}><Radio.Group options={[{ label: t('builder.wandbOnline'), value: 'online' }, { label: t('builder.wandbOffline'), value: 'offline' }]} /></Form.Item><Form.Item name={['wandb', 'project']} label={t('builder.wandbProject')} rules={[{ required: true, whitespace: true }]}><Input /></Form.Item><Form.Item name={['wandb', 'entity']} label={t('builder.wandbEntity')}><Input /></Form.Item><Form.Item name={['wandb', 'run_name']} label={t('builder.wandbRunName')}><Input /></Form.Item></div><Form.Item name={['wandb', 'tags']} label={t('builder.wandbTags')}><Select mode="tags" tokenSeparators={[',']} /></Form.Item><Form.Item name={['wandb', 'categories']} label={t('builder.wandbUploadContent')} rules={[{ required: true }]}><Checkbox.Group options={[{ label: t('evaluation.wandbSummary'), value: 'summary' }, { label: t('evaluation.wandbTasks'), value: 'tasks' }, { label: t('evaluation.wandbConfig'), value: 'config' }, { label: t('evaluation.wandbVideos'), value: 'videos' }]} /></Form.Item></> : null}
            </Card>
          </div> : null}
          {step === 3 ? <div className="evaluation-step">
            <Descriptions bordered column={1} size="small"><Descriptions.Item label={t('evaluation.name')}>{values?.name}</Descriptions.Item><Descriptions.Item label={t('evaluation.kindLabel')}>{t(`evaluation.kind.${evaluationKind}`)}</Descriptions.Item><Descriptions.Item label={t('evaluation.modelSource')}>{managedSource ? `${t('evaluation.managedDeployment')} · ${selectedDeployment?.name ?? values?.deployment_id}` : t('evaluation.temporaryCheckpoint')}</Descriptions.Item><Descriptions.Item label={t('evaluation.checkpoint')}>{managedSource ? selectedDeployment?.checkpoint_path : checkpointKind === 'indexed' ? evaluationKind === 'batch' ? values?.batch_checkpoint_ids?.join(', ') : values?.checkpoint_id : values?.checkpoint_path}</Descriptions.Item><Descriptions.Item label={t('evaluation.modelCombination')}>{effectiveCombinationId}</Descriptions.Item><Descriptions.Item label={t('evaluation.benchmark')}>{values?.benchmark_id} · {(evaluationKind === 'batch' ? values?.suites?.join(', ') : values?.suite) ?? values?.task_set ?? values?.split ?? t(`evaluation.preset.${preset}`)}</Descriptions.Item><Descriptions.Item label={t('evaluation.scale')}>{t(`evaluation.preset.${preset}`)}</Descriptions.Item><Descriptions.Item label={t('evaluation.gpuResources')}>{managedSource ? t('evaluation.reusedDeployment') : singleGpu ? `GPU ${visibleGpus[0]?.index}` : values?.resources?.strategy === 'fixed' ? `GPU ${values.resources.gpu_ids?.join(', ')}` : `${t('builder.autoGpu')} · ${requestedGpuCount}`}</Descriptions.Item></Descriptions>
            {preflightResult ? <div className="evaluation-preflight"><Alert type={preflightResult.ok ? 'success' : 'error'} showIcon message={preflightResult.ok ? t('evaluation.preflightPassed') : t('evaluation.preflightFailed')} />{preflightResult.items.map((item, index) => <Alert key={`${item.id ?? item.title}-${index}`} type={item.level === 'error' ? 'error' : item.level === 'warning' ? 'warning' : item.level === 'success' ? 'success' : 'info'} showIcon message={item.message_i18n?.[language] ?? item.message ?? t(`preflight.${item.id ?? 'unknown'}`)} />)}</div> : <Alert className="evaluation-preflight" type="info" showIcon message={t('evaluation.preflightRequired')} />}
          </div> : null}
        </Form>
      </AsyncState>
    </Modal>
  );
}

export function EvaluationsPage() {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [search, setSearch] = useState('');
  const [benchmark, setBenchmark] = useState<string>();
  const [status, setStatus] = useState<string>();
  const [selectedIds, setSelectedIds] = useState<React.Key[]>([]);
  const evaluations = useQuery({ queryKey: ['evaluations'], queryFn: () => api.evaluations.list(), refetchInterval: 5_000 });
  const groups = useQuery({ queryKey: ['evaluation-groups'], queryFn: () => api.evaluations.groups.list(), refetchInterval: 5_000 });
  const catalog = useQuery({ queryKey: ['evaluation-benchmarks'], queryFn: api.evaluations.benchmarks });
  const createOpen = searchParams.get('create') === '1';
  const rows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return (evaluations.data ?? []).filter((run) => (!term || [run.name, run.owner_name, run.benchmark_id, run.combination_id, run.checkpoint_path].join(' ').toLowerCase().includes(term)) && (!benchmark || run.benchmark_id === benchmark) && (!status || run.status === status));
  }, [benchmark, evaluations.data, search, status]);
  const groupRows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return (groups.data ?? []).filter((group) => (!term || [group.name, group.owner_name, group.kind, ...group.runs.map((run) => `${run.benchmark_id} ${run.checkpoint_path ?? ''}`)].join(' ').toLowerCase().includes(term)) && (!benchmark || group.runs.some((run) => run.benchmark_id === benchmark)) && (!status || group.status === status));
  }, [benchmark, groups.data, search, status]);
  const selectedRuns = selectedIds.flatMap((id) => evaluations.data?.find((run) => run.id === String(id)) ?? []);
  const compareIssue = comparisonIssue(selectedRuns);
  const closeCreate = () => setSearchParams((current) => { const next = new URLSearchParams(current); ['create', 'checkpoint_id', 'checkpoint'].forEach((key) => next.delete(key)); return next; }, { replace: true });

  return <div className="page evaluations-page">
    <PageIntro title={t('evaluation.title')} subtitle={t('evaluation.subtitle')} actions={<Space><Button icon={<BarChartOutlined />} disabled={Boolean(compareIssue)} onClick={() => navigate(`/evaluations/compare?ids=${selectedIds.map(String).join(',')}`)}>{t('evaluation.compare')} {selectedIds.length ? `(${selectedIds.length})` : ''}</Button><Button type="primary" icon={<PlusOutlined />} onClick={() => setSearchParams((current) => { const next = new URLSearchParams(current); next.set('create', '1'); return next; })}>{t('evaluation.create')}</Button></Space>} />
    {selectedIds.length ? <Alert className="comparison-hint" type={compareIssue ? 'warning' : 'info'} showIcon message={compareIssue ? t(`evaluation.compareIssue.${compareIssue}`) : t('evaluation.compareReady', { count: selectedIds.length })} /> : null}
    <Card>
      <div className="table-toolbar evaluation-toolbar"><Input className="table-search" allowClear prefix={<SearchOutlined />} value={search} onChange={(event) => setSearch(event.target.value)} placeholder={t('common.search')} /><Space wrap><Select allowClear value={benchmark} onChange={setBenchmark} placeholder={t('evaluation.benchmark')} options={(catalog.data?.benchmarks ?? []).map((item) => ({ value: item.id, label: localized(language, item.name, item.name_zh, item.id) }))} /><Select allowClear value={status} onChange={setStatus} placeholder={t('common.status')} options={['queued', 'starting', 'running', 'stopping', 'partial', 'completed', 'failed', 'stopped', 'cancelled', 'interrupted'].map((value) => ({ value, label: t(`status.${value}`, { defaultValue: value }) }))} /><Button icon={<ExperimentOutlined />} onClick={() => { void evaluations.refetch(); void groups.refetch(); }}>{t('common.refresh')}</Button></Space></div>
      {groupRows.length || groups.isLoading || groups.error ? <div className="evaluation-group-list"><Typography.Title level={4}>{t('evaluation.groupedEvaluations')}</Typography.Title><AsyncState loading={groups.isLoading} error={groups.error} onRetry={() => void groups.refetch()}><Table<EvaluationGroup> rowKey="id" pagination={false} dataSource={groupRows} scroll={{ x: 900 }} onRow={(group) => ({ onClick: () => navigate(`/evaluation-groups/${group.id}`) })} columns={[
        { title: t('evaluation.name'), dataIndex: 'name', render: (value: string, group) => <div><Typography.Text strong>{value}</Typography.Text><small className="table-subtitle">{group.owner_name ?? '—'} · {group.runs.length} {t('evaluation.childRuns')}</small></div> },
        { title: t('evaluation.kindLabel'), dataIndex: 'kind', render: (value: EvaluationKind) => <Tag color="blue">{t(`evaluation.kind.${value}`, { defaultValue: value })}</Tag> },
        { title: t('common.status'), dataIndex: 'status', render: (value) => <StatusTag status={value} /> },
        { title: t('evaluation.successRate'), render: (_, group) => rateFromSummary(group.result?.summary ?? group.result_summary) },
        { title: t('evaluation.completedRuns'), render: (_, group) => `${Number(group.result_summary.num_completed ?? group.runs.filter((run) => run.status === 'completed').length)} / ${group.runs.length}` },
        { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm') : '—' },
        { title: t('common.actions'), render: (_, group) => <Button size="small" onClick={(event) => { event.stopPropagation(); navigate(`/evaluation-groups/${group.id}`); }}>{t('common.view')}</Button> },
      ]} /></AsyncState></div> : null}
      <Typography.Title level={4}>{t('evaluation.individualRuns')}</Typography.Title>
      <AsyncState loading={evaluations.isLoading} error={evaluations.error} empty={!rows.length} onRetry={() => void evaluations.refetch()}><Table<EvaluationRun> rowKey="id" dataSource={rows} scroll={{ x: 1100 }} rowSelection={{ selectedRowKeys: selectedIds, onChange: setSelectedIds, preserveSelectedRowKeys: true, getCheckboxProps: (run) => ({ disabled: run.status !== 'completed' }) }} onRow={(run) => ({ onClick: () => navigate(`/evaluations/${run.id}`) })} columns={[
        { title: t('evaluation.name'), dataIndex: 'name', fixed: 'left', render: (value: string, run) => <div><Typography.Text strong>{value}</Typography.Text><small className="table-subtitle">{run.owner_name ?? '—'}</small></div> },
        { title: t('common.status'), dataIndex: 'status', render: (value) => <StatusTag status={value} /> },
        { title: t('evaluation.benchmark'), dataIndex: 'benchmark_id', render: (value: string, run) => <div><Typography.Text>{value}</Typography.Text><small className="table-subtitle">{run.suite ?? run.task_set ?? run.split ?? run.preset}</small></div> },
        { title: t('evaluation.checkpoint'), render: (_, run) => <Typography.Text code className="table-wrap-text">{run.checkpoint_path?.split('/').pop() ?? run.checkpoint_id ?? '—'}</Typography.Text> },
        { title: t('evaluation.successRate'), render: (_, run) => <Typography.Text strong className={run.result ? 'success' : undefined}>{successRate(run)}</Typography.Text> },
        { title: t('evaluation.gpuResources'), render: (_, run) => run.assigned_gpu_ids.length ? run.assigned_gpu_ids.map((id) => <span className="gpu-chip" key={id}>{id}</span>) : run.queue_position ? `#${run.queue_position}` : '1 GPU' },
        { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm') : '—' },
        { title: t('common.actions'), fixed: 'right', render: (_, run) => <Button size="small" onClick={(event) => { event.stopPropagation(); navigate(`/evaluations/${run.id}`); }}>{t('common.view')}</Button> },
      ]} /></AsyncState>
    </Card>
    {createOpen ? <CreateEvaluationModal open onClose={closeCreate} onCreated={async (result) => { closeCreate(); await Promise.all([queryClient.invalidateQueries({ queryKey: ['evaluations'] }), queryClient.invalidateQueries({ queryKey: ['evaluation-groups'] })]); navigate(result.group ? `/evaluation-groups/${result.group.id}` : `/evaluations/${result.evaluation.id}`); }} /> : null}
  </div>;
}
