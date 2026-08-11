import type {
  Capability,
  AuditEvent,
  CapabilityKind,
  CapabilityResponse,
  Checkpoint,
  CheckpointDetail,
  DashboardData,
  DatasetValidationResult,
  DatasetMixture,
  DatasetInspectionResult,
  DatasetPreview,
  DatasetRegistration,
  DeploymentAdapter,
  DeploymentCapabilities,
  DeploymentCombination,
  DeploymentCreateRequest,
  DeploymentMutationResult,
  DeploymentStatus,
  DirectoryListing,
  EvaluationBenchmark,
  EvaluationCapabilities,
  EvaluationCheckpointInspection,
  EvaluationCheckpointSource,
  EvaluationComparison,
  EvaluationCreateResult,
  EvaluationCreateRequest,
  EvaluationArtifact,
  EvaluationGroup,
  EvaluationMatrix,
  EvaluationPreset,
  EvaluationResult,
  EvaluationRun,
  EvaluationSeries,
  EvaluationStatus,
  EvaluationVideo,
  EvaluationWandbConfig,
  Experiment,
  ExperimentSpec,
  GPU,
  GPUInventory,
  InferenceRun,
  Job,
  JobStatus,
  ModelDeployment,
  ModelPublication,
  ParameterSchema,
  PlaygroundInferenceInput,
  Paged,
  PreflightItem,
  PreflightResult,
  ReferenceResultsMatch,
  ReferenceResultsSnapshot,
  RemoteServerMetrics,
  RegistryBrowseResponse,
  RegistryOverlayPreview,
  RegistryOverlayState,
  ResolveResult,
  ResourceRecord,
  SetupStatus,
  StorageSummary,
  SystemMetrics,
  SystemSettings,
  Template,
  TrainingTarget,
  TemplateDuplicateDetail,
  TemplateDuplicateMatch,
  User,
  UtilityRun,
  WandbCategoryCapability,
  WandbRunConfig,
  WandbSecretStatus,
  WandbUploadCategory,
  Workload,
  WorkflowSchema,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api/v1';

export class ApiError extends Error {
  readonly status: number;
  readonly detail?: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function getCookie(name: string): string | undefined {
  const encoded = `${encodeURIComponent(name)}=`;
  return document.cookie
    .split('; ')
    .find((item) => item.startsWith(encoded))
    ?.slice(encoded.length);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = init.method?.toUpperCase() ?? 'GET';
  const headers = new Headers(init.headers);
  headers.set('Accept', 'application/json');
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    const csrfToken = getCookie('csrf_token') ?? getCookie('alphabrain_csrf');
    if (csrfToken) headers.set('X-CSRF-Token', decodeURIComponent(csrfToken));
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      cache: method === 'GET' ? 'no-store' : undefined,
      ...init,
      headers,
      credentials: 'include',
    });
  } catch (error) {
    throw new ApiError(error instanceof Error ? error.message : 'Network request failed', 0, error);
  }

  if (response.status === 204) return undefined as T;
  const contentType = response.headers.get('content-type') ?? '';
  const payload: unknown = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const record = isRecord(payload) ? payload : undefined;
    const detail = record?.detail ?? record?.message ?? payload;
    const detailRecord = isRecord(detail) ? detail : undefined;
    const message = typeof detail === 'string'
      ? detail
      : stringValue(detailRecord?.message)
        || stringValue(detailRecord?.code)
        || response.statusText
        || 'Request failed';
    throw new ApiError(message, response.status, detail);
  }
  return payload as T;
}

function body(value: unknown): string {
  return JSON.stringify(value);
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function listFrom<T>(value: T[] | Paged<T> | { data?: T[] } | undefined | null): T[] {
  if (Array.isArray(value)) return value;
  if (value && 'items' in value && Array.isArray(value.items)) return value.items;
  if (value && 'data' in value && Array.isArray(value.data)) return value.data;
  return [];
}

function stringValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function numberValue(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function recordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function recordList(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) return recordArray(value);
  if (!isRecord(value)) return [];
  return recordArray(value.items).length ? recordArray(value.items) : recordArray(value.data);
}

const WANDB_CATEGORY_IDS = new Set<WandbUploadCategory>(['metrics', 'config', 'system', 'checkpoints', 'videos']);

function normalizeWandbRunConfig(value: unknown): WandbRunConfig | undefined {
  if (!isRecord(value)) return undefined;
  const rawCategories = Array.isArray(value.categories) ? value.categories : ['metrics', 'config', 'system'];
  const categories = Array.from(new Set(rawCategories.filter(
    (item): item is WandbUploadCategory => typeof item === 'string' && WANDB_CATEGORY_IDS.has(item as WandbUploadCategory),
  )));
  return {
    enabled: Boolean(value.enabled),
    mode: value.mode === 'offline' || value.mode === 'disabled' ? value.mode : 'online',
    project: stringValue(value.project, 'AlphaBrain'),
    entity: stringValue(value.entity),
    run_name: stringValue(value.run_name),
    group: stringValue(value.group),
    job_type: stringValue(value.job_type),
    tags: Array.isArray(value.tags) ? value.tags.filter((item): item is string => typeof item === 'string') : [],
    notes: stringValue(value.notes),
    categories,
  };
}

function wandbPayload(value: WandbRunConfig): Record<string, unknown> {
  return {
    enabled: Boolean(value.enabled),
    mode: value.mode,
    project: value.project,
    entity: value.entity,
    run_name: value.run_name,
    group: value.group,
    job_type: value.job_type,
    tags: [...value.tags],
    notes: value.notes,
    categories: [...value.categories],
  };
}

function normalizeUser(value: unknown, experimentalAvailable?: boolean, deploymentMode?: unknown): User {
  const row = isRecord(value) ? value : {};
  return {
    id: stringValue(row.id),
    username: stringValue(row.username),
    display_name: stringValue(row.display_name) || undefined,
    role: row.role === 'administrator' ? 'administrator' : 'researcher',
    active: row.is_active !== false,
    locale: row.language === 'en-US' ? 'en-US' : 'zh-CN',
    theme: row.theme === 'dark' ? 'dark' : 'light',
    gpu_refresh_interval_seconds: numberValue(row.gpu_refresh_interval_seconds, 5),
    experimental_enabled: Boolean(row.experimental_enabled),
    experimental_available: experimentalAvailable,
    deployment_mode: deploymentMode === 'lab' ? 'laboratory' : deploymentMode === 'personal' ? 'personal' : undefined,
    created_at: stringValue(row.created_at) || undefined,
  };
}

function normalizeJob(value: unknown): Job {
  const row = isRecord(value) ? value : {};
  const assigned = Array.isArray(row.assigned_gpu_ids) ? row.assigned_gpu_ids.map(Number) : [];
  const requested = Array.isArray(row.requested_gpu_ids) ? row.requested_gpu_ids.map(Number) : [];
  const id = stringValue(row.id);
  return {
    id,
    experiment_id: stringValue(row.experiment_id) || undefined,
    name: stringValue(row.name) || `Job ${id.slice(0, 8)}`,
    owner_id: stringValue(row.owner_id) || undefined,
    owner_name: stringValue(row.owner_name) || undefined,
    status: stringValue(row.status, 'queued') as JobStatus,
    queue_position: numberValue(row.queue_position) || undefined,
    gpu_ids: assigned.length ? assigned : requested,
    stage_name: stringValue(row.stage_name) || stringValue(row.phase) || undefined,
    command: stringValue(row.command_preview) || stringValue(row.command) || undefined,
    pid: typeof row.pid === 'number' ? row.pid : undefined,
    progress: typeof row.progress === 'number' ? row.progress : undefined,
    created_at: stringValue(row.queued_at) || stringValue(row.created_at) || undefined,
    started_at: stringValue(row.started_at) || undefined,
    finished_at: stringValue(row.finished_at) || undefined,
    error_summary: stringValue(row.error) || stringValue(row.error_summary) || undefined,
    latest_metrics: isRecord(row.latest_metrics) ? row.latest_metrics as Record<string, number> : undefined,
  };
}

function normalizeExperimentSpec(value: unknown): ExperimentSpec {
  const row = isRecord(value) ? value : {};
  const architecture = isRecord(row.architecture) ? row.architecture : {};
  const training = isRecord(row.training) ? row.training : {};
  const dataset = isRecord(row.dataset) ? row.dataset : {};
  const metadata = isRecord(row.metadata) ? row.metadata : {};
  const rawParameters = isRecord(row.parameters) ? row.parameters : {};
  const parameters: Record<string, unknown> = { ...rawParameters };
  const runName = stringValue(rawParameters.run_id);
  const checkpointPath = stringValue(training.checkpoint) || stringValue(rawParameters.pretrained_checkpoint) || undefined;
  delete parameters.run_id;
  delete parameters.pretrained_checkpoint;
  if ('per_device_batch_size' in parameters) {
    parameters.batch_size = parameters.per_device_batch_size;
    delete parameters.per_device_batch_size;
  }
  if ('max_train_steps' in parameters) {
    parameters.max_steps = parameters.max_train_steps;
    delete parameters.max_train_steps;
  }
  const resources = isRecord(row.resources) ? row.resources : {};
  const experimental = isRecord(row.experimental) ? row.experimental : {};
  const rawWorkflow = isRecord(row.workflow) ? row.workflow : {};
  const rawResume = isRecord(row.resume) ? row.resume : {};
  const legacyResumeCheckpoint = stringValue(training.resume_checkpoint) || stringValue(rawParameters.resume_checkpoint);
  const resumeMode = rawResume.mode === 'weights_only' || rawResume.mode === 'full_state'
    ? rawResume.mode
    : Boolean(training.is_resume) && (legacyResumeCheckpoint || checkpointPath) ? 'full_state' : 'none';
  const resumeCheckpoint = stringValue(rawResume.checkpoint)
    || legacyResumeCheckpoint
    || (resumeMode === 'full_state' && Boolean(training.is_resume) ? checkpointPath : undefined);
  return {
    spec_version: 2,
    name: runName,
    description: stringValue(metadata.description) || undefined,
    backbone_id: stringValue(architecture.backbone) || undefined,
    action_head_id: stringValue(architecture.action_head) || undefined,
    method_id: stringValue(training.method) || undefined,
    dataset_id: stringValue(dataset.id) || undefined,
    dataset_path: stringValue(dataset.root) || stringValue(dataset.data_root) || undefined,
    dataset_registration_id: stringValue(dataset.registration_id) || undefined,
    dataset_mixture_id: stringValue(dataset.mixture_id) || undefined,
    checkpoint_path: checkpointPath,
    workflow: stringValue(rawWorkflow.id) ? {
      id: stringValue(rawWorkflow.id),
      schema_version: numberValue(rawWorkflow.schema_version, 1),
      config: isRecord(rawWorkflow.config) ? rawWorkflow.config : {},
    } : undefined,
    resume: { mode: resumeMode, ...(resumeCheckpoint ? { checkpoint: resumeCheckpoint } : {}) },
    parameters,
    resources: {
      strategy: resources.allocation === 'fixed' ? 'fixed' : 'auto',
      gpu_count: numberValue(resources.num_gpus, 1),
      gpu_ids: Array.isArray(resources.gpu_ids) ? resources.gpu_ids.map(Number) : undefined,
    },
    wandb: normalizeWandbRunConfig(row.wandb),
    expert_overrides: isRecord(row.expert_overrides) ? row.expert_overrides : {},
    experimental_confirmed: Boolean(experimental.risk_acknowledged),
  };
}

function toBackendSpec(spec: ExperimentSpec): Record<string, unknown> {
  const parameters = { ...spec.parameters };
  parameters.run_id = spec.name;
  if ('batch_size' in parameters) {
    parameters.per_device_batch_size = parameters.batch_size;
    delete parameters.batch_size;
  }
  if ('max_steps' in parameters) {
    parameters.max_train_steps = parameters.max_steps;
    delete parameters.max_steps;
  }
  if (spec.checkpoint_path) parameters.pretrained_checkpoint = spec.checkpoint_path;
  return {
    spec_version: 2,
    architecture: { backbone: spec.backbone_id, action_head: spec.action_head_id },
    training: { method: spec.method_id, ...(spec.checkpoint_path ? { checkpoint: spec.checkpoint_path } : {}) },
    dataset: {
      id: spec.dataset_id,
      ...(spec.dataset_path ? { root: spec.dataset_path, data_root: spec.dataset_path } : {}),
      ...(spec.dataset_registration_id ? { registration_id: spec.dataset_registration_id } : {}),
      ...(spec.dataset_mixture_id ? { mixture_id: spec.dataset_mixture_id } : {}),
    },
    parameters,
    resources: {
      allocation: spec.resources.strategy,
      num_gpus: spec.resources.gpu_count,
      gpu_ids: spec.resources.strategy === 'fixed' ? (spec.resources.gpu_ids ?? []).map(Number) : [],
    },
    ...(spec.workflow ? { workflow: spec.workflow } : {}),
    resume: spec.resume ?? { mode: 'none', checkpoint: null },
    ...(spec.wandb ? { wandb: wandbPayload(spec.wandb) } : {}),
    expert_overrides: spec.expert_overrides ?? {},
    experimental: {
      enabled: Boolean(spec.experimental_confirmed),
      risk_acknowledged: Boolean(spec.experimental_confirmed),
    },
    metadata: { description: spec.description ?? '', source_template_id: spec.template_id ?? null },
  };
}

function experimentRequest(spec: ExperimentSpec): Record<string, unknown> {
  return {
    name: spec.name,
    spec: toBackendSpec(spec),
    acknowledge_experimental: Boolean(spec.experimental_confirmed),
  };
}

function normalizeExperiment(value: unknown): Experiment {
  const row = isRecord(value) ? value : {};
  const spec = normalizeExperimentSpec(row.spec);
  const stages = recordArray(row.stages).map((stage) => {
    const jobs = recordArray(stage.jobs).map(normalizeJob);
    return {
      id: stringValue(stage.id),
      name: stringValue(stage.name) || stringValue(stage.phase),
      order: numberValue(stage.position),
      status: jobs[0]?.status ?? 'blocked' as JobStatus,
      job_id: jobs[0]?.id,
    };
  });
  return {
    id: stringValue(row.id),
    name: stringValue(row.name),
    description: spec.description,
    owner_id: stringValue(row.owner_id) || undefined,
    owner_name: stringValue(row.owner_name) || undefined,
    architecture: spec.backbone_id,
    method: spec.method_id || stringValue(row.family),
    dataset: spec.dataset_id,
    status: stringValue(row.status, 'draft') as JobStatus,
    created_at: stringValue(row.created_at) || undefined,
    updated_at: stringValue(row.updated_at) || undefined,
    job_id: stages.flatMap((stage) => stage.job_id ?? [])[0],
    stages,
    config: isRecord(row.resolved) ? row.resolved : {},
  };
}

function normalizeTemplate(value: unknown): Template {
  const row = isRecord(value) ? value : {};
  const spec = normalizeExperimentSpec(row.spec);
  const nameI18n = isRecord(row.name_i18n) ? row.name_i18n : {};
  const descriptionI18n = isRecord(row.description_i18n) ? row.description_i18n : {};
  return {
    id: stringValue(row.id),
    name: stringValue(row.name),
    description: stringValue(row.description) || undefined,
    visibility: row.visibility === 'shared' ? 'shared' : 'personal',
    owner_id: stringValue(row.owner_id) || undefined,
    owner_name: stringValue(row.owner_name) || undefined,
    architecture: spec.backbone_id,
    method: spec.method_id,
    dataset: spec.dataset_id,
    version: numberValue(row.version, 1),
    updated_at: stringValue(row.updated_at) || undefined,
    spec: { ...spec, name: spec.name || stringValue(row.name), description: stringValue(row.description) || undefined },
    builtin: Boolean(row.builtin),
    name_i18n: { 'zh-CN': stringValue(nameI18n['zh-CN']) || undefined, 'en-US': stringValue(nameI18n['en-US']) || undefined },
    description_i18n: { 'zh-CN': stringValue(descriptionI18n['zh-CN']) || undefined, 'en-US': stringValue(descriptionI18n['en-US']) || undefined },
    category: stringValue(row.category) || undefined,
    tags: Array.isArray(row.tags) ? row.tags.map(String) : undefined,
    availability: row.availability === 'ready' || row.availability === 'missing' ? row.availability : row.availability === 'partial' ? 'partial' : undefined,
    requirements: recordArray(row.requirements).map((item) => ({ id: stringValue(item.id), path: stringValue(item.path) || undefined, ready: Boolean(item.ready) })),
    recommended_gpu_count: typeof row.recommended_gpu_count === 'number' ? row.recommended_gpu_count : undefined,
  };
}

export function templateDuplicateDetail(error: unknown): TemplateDuplicateDetail | undefined {
  if (!(error instanceof ApiError) || error.status !== 409 || !isRecord(error.detail)) return undefined;
  if (error.detail.code !== 'duplicate_template' || !Array.isArray(error.detail.matches)) return undefined;
  const matches = error.detail.matches.flatMap((value): TemplateDuplicateMatch[] => {
    if (!isRecord(value) || !stringValue(value.id) || !stringValue(value.name)) return [];
    return [{
      id: stringValue(value.id),
      name: stringValue(value.name),
      owner_id: stringValue(value.owner_id) || undefined,
      owner_name: stringValue(value.owner_name) || undefined,
      visibility: value.visibility === 'shared' ? 'shared' : 'personal',
      builtin: Boolean(value.builtin),
      same_name: Boolean(value.same_name),
      same_spec: Boolean(value.same_spec),
    }];
  });
  return matches.length ? { code: 'duplicate_template', matches } : undefined;
}

function normalizeCheckpoint(value: unknown): Checkpoint {
  const row = isRecord(value) ? value : {};
  const nameI18n = isRecord(row.name_i18n) ? row.name_i18n : {};
  const descriptionI18n = isRecord(row.description_i18n) ? row.description_i18n : {};
  const missingRequirements = isRecord(row.missing_requirements_i18n) ? row.missing_requirements_i18n : {};
  const inspectionSummary = isRecord(row.inspection_summary) ? row.inspection_summary : {};
  const checkpointFormatLabel = isRecord(row.checkpoint_format_label_i18n)
    ? row.checkpoint_format_label_i18n
    : isRecord(inspectionSummary.checkpoint_format_label) ? inspectionSummary.checkpoint_format_label : {};
  return {
    id: stringValue(row.id),
    experiment_id: stringValue(row.experiment_id) || undefined,
    experiment_name: stringValue(row.experiment_name) || undefined,
    owner_name: stringValue(row.owner_name) || undefined,
    path: stringValue(row.path),
    step: typeof row.step === 'number' ? row.step : undefined,
    size_bytes: typeof row.size_bytes === 'number' ? row.size_bytes : undefined,
    kind: stringValue(row.kind) || undefined,
    created_at: stringValue(row.created_at) || undefined,
    complete: row.is_complete !== false,
    resumable: Boolean(row.is_resumable),
    important: Boolean(row.important),
    architecture_fingerprint: stringValue(row.architecture_fingerprint) || undefined,
    best_score: typeof row.best_score === 'number' ? row.best_score : undefined,
    can_delete: Boolean(row.can_delete),
    can_package: Boolean(row.can_package),
    deployable: typeof row.can_deploy === 'boolean'
      ? row.can_deploy
      : typeof row.deployable === 'boolean' ? row.deployable : undefined,
    builtin: Boolean(row.builtin),
    name_i18n: { 'zh-CN': stringValue(nameI18n['zh-CN']) || undefined, 'en-US': stringValue(nameI18n['en-US']) || undefined },
    combination_id: stringValue(row.combination_id) || undefined,
    description: stringValue(row.description) || undefined,
    description_i18n: { 'zh-CN': stringValue(descriptionI18n['zh-CN']) || undefined, 'en-US': stringValue(descriptionI18n['en-US']) || undefined },
    missing_requirements_i18n: {
      'zh-CN': Array.isArray(missingRequirements['zh-CN']) ? missingRequirements['zh-CN'].map(String) : undefined,
      'en-US': Array.isArray(missingRequirements['en-US']) ? missingRequirements['en-US'].map(String) : undefined,
    },
    checkpoint_family: stringValue(row.checkpoint_family) || stringValue(inspectionSummary.checkpoint_family) || undefined,
    checkpoint_format: stringValue(row.checkpoint_format) || stringValue(inspectionSummary.checkpoint_format) || undefined,
    checkpoint_format_label_i18n: {
      'zh-CN': stringValue(checkpointFormatLabel['zh-CN']) || undefined,
      'en-US': stringValue(checkpointFormatLabel['en-US']) || undefined,
    },
    inspection_summary: {
      format: stringValue(inspectionSummary.format) || undefined,
      checkpoint_family: stringValue(inspectionSummary.checkpoint_family) || undefined,
      checkpoint_format: stringValue(inspectionSummary.checkpoint_format) || undefined,
      checkpoint_format_label: {
        'zh-CN': stringValue(checkpointFormatLabel['zh-CN']) || undefined,
        'en-US': stringValue(checkpointFormatLabel['en-US']) || undefined,
      },
      framework: stringValue(inspectionSummary.framework) || undefined,
      combination_id: stringValue(inspectionSummary.combination_id) || undefined,
      issue_codes: Array.isArray(inspectionSummary.issue_codes) ? inspectionSummary.issue_codes.map(String) : undefined,
    },
  };
}

function normalizeCheckpointDetail(value: unknown): CheckpointDetail {
  const row = isRecord(value) ? value : {};
  const tools = isRecord(row.tools) ? row.tools : {};
  const publish = isRecord(tools.publish_huggingface) ? tools.publish_huggingface : {};
  const merge = isRecord(tools.merge_lora) ? tools.merge_lora : {};
  const special = isRecord(tools.add_qwen_special_tokens) ? tools.add_qwen_special_tokens : {};
  return {
    ...normalizeCheckpoint(row),
    name: stringValue(row.name) || undefined,
    job_id: stringValue(row.job_id) || undefined,
    complete: row.complete !== false,
    resumable: Boolean(row.resumable),
    metadata: isRecord(row.metadata) ? row.metadata : {},
    inspection: isRecord(row.inspection) ? row.inspection : {},
    tools: {
      publish_huggingface: { available: Boolean(publish.available) },
      merge_lora: {
        available: Boolean(merge.available),
        reason: stringValue(merge.reason) || undefined,
        adapter_path: stringValue(merge.adapter_path) || undefined,
        action_model_path: stringValue(merge.action_model_path) || undefined,
        suggested_output_path: stringValue(merge.suggested_output_path) || undefined,
        output_exists: Boolean(merge.output_exists),
        models: Array.isArray(merge.models) ? merge.models.map(String) : undefined,
        experimental: Boolean(merge.experimental),
      },
      add_qwen_special_tokens: {
        available: Boolean(special.available),
        reason: stringValue(special.reason) || undefined,
      },
    },
  };
}

function normalizeGPU(value: unknown): GPU {
  const row = isRecord(value) ? value : {};
  const processes = Array.isArray(row.processes) ? row.processes : [];
  return {
    id: stringValue(row.uuid) || undefined,
    index: numberValue(row.index),
    name: stringValue(row.name, 'GPU'),
    memory_used_mb: numberValue(row.memory_used_bytes) / 1024 ** 2,
    memory_total_mb: numberValue(row.memory_total_bytes) / 1024 ** 2,
    utilization_percent: numberValue(row.utilization_percent),
    temperature_c: typeof row.temperature_c === 'number' ? row.temperature_c : undefined,
    available: Boolean(row.available),
    external_processes: processes.length,
    error: stringValue(row.error) || undefined,
    job: row.reserved_by_job_id ? {
      id: stringValue(row.reserved_by_job_id),
      name: `Job ${stringValue(row.reserved_by_job_id).slice(0, 8)}`,
      status: 'running',
    } : undefined,
  };
}

function normalizeGPUInventory(value: unknown): GPUInventory {
  const row = isRecord(value) ? value : {};
  return {
    available: Boolean(row.available),
    error: stringValue(row.error) || undefined,
    items: recordArray(row.items).map(normalizeGPU),
  };
}

export function healthyGpus(value: GPUInventory | undefined | null): GPU[] {
  return value?.items.filter((gpu) => !gpu.error) ?? [];
}

function normalizeStorage(value: unknown): StorageSummary | undefined {
  const row = isRecord(value) ? value : undefined;
  if (!row) return undefined;
  return {
    path: stringValue(row.path),
    mount_point: stringValue(row.mount_point) || undefined,
    used_bytes: numberValue(row.used_bytes),
    total_bytes: numberValue(row.total_bytes),
    free_bytes: numberValue(row.free_bytes),
    warning: Boolean(row.low_space),
  };
}

function normalizeSystemMetrics(value: unknown): SystemMetrics | undefined {
  const row = isRecord(value) ? value : undefined;
  if (!row) return undefined;
  const load = isRecord(row.load) ? row.load : undefined;
  const loadArray = Array.isArray(row.load) ? row.load : undefined;
  const memory = isRecord(row.memory) ? row.memory : undefined;
  const error = isRecord(row.error) ? row.error : undefined;
  const normalizedLoad = load || loadArray ? {
    one_minute: numberValue(load?.one_minute ?? load?.['1m'] ?? loadArray?.[0]),
    five_minutes: numberValue(load?.five_minutes ?? load?.['5m'] ?? loadArray?.[1]),
    fifteen_minutes: numberValue(load?.fifteen_minutes ?? load?.['15m'] ?? loadArray?.[2]),
  } : undefined;
  const normalizedMemory = memory ? {
    total_bytes: numberValue(memory.total_bytes ?? memory.total),
    used_bytes: numberValue(memory.used_bytes ?? memory.used),
    available_bytes: numberValue(memory.available_bytes ?? memory.available),
    percent: numberValue(memory.percent),
  } : undefined;
  return {
    available: row.available !== false && Boolean(
      typeof row.cpu_percent === 'number' || normalizedLoad || normalizedMemory,
    ),
    cpu_percent: typeof row.cpu_percent === 'number' ? numberValue(row.cpu_percent) : undefined,
    load: normalizedLoad,
    memory: normalizedMemory,
    error: error ? {
      code: stringValue(error.code, 'system_metrics_unavailable'),
      message: stringValue(error.message) || undefined,
    } : undefined,
  };
}

function normalizeRemoteServerMetrics(value: unknown): RemoteServerMetrics {
  const row = isRecord(value) ? value : {};
  const error = isRecord(row.error) ? row.error : undefined;
  return {
    enabled: Boolean(row.enabled),
    available: Boolean(row.available),
    target: stringValue(row.target) || undefined,
    hostname: stringValue(row.hostname) || undefined,
    collected_at: stringValue(row.collected_at) || undefined,
    stale: Boolean(row.stale),
    gpus: recordArray(row.gpus).map(normalizeGPU),
    system_metrics: normalizeSystemMetrics(row.system_metrics),
    storage: normalizeStorage(row.storage),
    gpu_error: stringValue(row.gpu_error) || undefined,
    error: error ? {
      code: stringValue(error.code, 'remote_metrics_unavailable'),
      message: stringValue(error.message) || undefined,
    } : undefined,
  };
}

function capabilityFrom(
  raw: Record<string, unknown>,
  kind: CapabilityKind,
  combinations: CapabilityResponse['combinations'],
): Capability {
  const id = stringValue(raw.id);
  const labels = isRecord(raw.label) ? raw.label : {};
  const descriptions = isRecord(raw.description) ? raw.description : {};
  const related = (combinations ?? []).filter((combo) =>
    (kind === 'backbone' && combo.backbone === id)
    || (kind === 'action_head' && combo.action_head === id)
    || (kind === 'method' && combo.method === id)
    || (kind === 'dataset' && combo.datasets.includes(id)),
  );
  const compatibleIds = new Set<string>();
  related.forEach((combo) => {
    compatibleIds.add(combo.backbone);
    compatibleIds.add(combo.action_head);
    compatibleIds.add(combo.method);
    combo.datasets.forEach((dataset) => compatibleIds.add(dataset));
  });
  compatibleIds.delete(id);
  const minGpus = related.map((combo) => combo.min_gpus).filter((value): value is number => typeof value === 'number');
  const parameters = normalizeParameters(raw.parameters);
  const pretrainedDirectory = isRecord(raw.pretrained_directory) ? raw.pretrained_directory : undefined;
  const pretrainedIssue = pretrainedDirectory && isRecord(pretrainedDirectory.issue)
    ? pretrainedDirectory.issue
    : undefined;
  const issueMessages = pretrainedIssue && isRecord(pretrainedIssue.message_i18n)
    ? pretrainedIssue.message_i18n
    : undefined;
  return {
    id,
    kind,
    name: stringValue(raw.name) || stringValue(labels['en-US']) || id,
    name_zh: stringValue(raw.name_zh) || stringValue(labels['zh-CN']) || undefined,
    description: (typeof raw.description === 'string' ? raw.description : stringValue(descriptions['en-US'])) || undefined,
    description_zh: stringValue(raw.description_zh) || stringValue(descriptions['zh-CN']) || undefined,
    status: raw.status === 'experimental' ? 'experimental' : raw.status === 'unsupported' ? 'unsupported' : 'verified',
    category: kind === 'backbone' ? stringValue(raw.category) || stringValue(raw.kind) || undefined : undefined,
    compatible_with: Array.from(compatibleIds),
    recommended_gpu_count: typeof raw.recommended_gpu_count === 'number'
      ? raw.recommended_gpu_count
      : minGpus.length ? Math.min(...minGpus) : undefined,
    requires_checkpoint: Boolean(raw.requires_checkpoint),
    pretrained_directory: pretrainedDirectory ? {
      required: Boolean(pretrainedDirectory.required),
      configured: Boolean(pretrainedDirectory.configured),
      path: stringValue(pretrainedDirectory.path) || undefined,
      exists: Boolean(pretrainedDirectory.exists),
      issue: pretrainedIssue ? {
        code: stringValue(pretrainedIssue.code, 'pretrained_directory_unavailable'),
        variables: Array.isArray(pretrainedIssue.variables) ? pretrainedIssue.variables.map(String) : undefined,
        message_i18n: issueMessages ? {
          ...(stringValue(issueMessages['zh-CN']) ? { 'zh-CN': stringValue(issueMessages['zh-CN']) } : {}),
          ...(stringValue(issueMessages['en-US']) ? { 'en-US': stringValue(issueMessages['en-US']) } : {}),
        } : undefined,
      } : undefined,
    } : undefined,
    parameters,
  };
}

function normalizeParameters(value: unknown): ParameterSchema[] {
  const parameterTypes = new Set<ParameterSchema['type']>(['string', 'number', 'integer', 'boolean', 'select', 'path', 'password', 'textarea']);
  return recordArray(value).flatMap((parameter): ParameterSchema[] => {
    const key = stringValue(parameter.key);
    if (!key) return [];
    const rawType = stringValue(parameter.type, 'string') as ParameterSchema['type'];
    const type = parameterTypes.has(rawType) ? rawType : 'string';
    const options = recordArray(parameter.options).flatMap((option) => {
      const value = option.value;
      if (typeof value !== 'string' && typeof value !== 'number') return [];
      const optionLabels = isRecord(option.labels) ? option.labels : {};
      return [{
        label: stringValue(option.label) || stringValue(optionLabels['en-US']) || String(value),
        label_zh: stringValue(option.label_zh) || stringValue(optionLabels['zh-CN']) || undefined,
        value,
      }];
    });
    const labels = isRecord(parameter.label) ? parameter.label : isRecord(parameter.labels) ? parameter.labels : {};
    const descriptions = isRecord(parameter.description) ? parameter.description : {};
    return [{
      key,
      type,
      label: (typeof parameter.label === 'string' ? parameter.label : stringValue(labels['en-US'])) || undefined,
      label_zh: stringValue(parameter.label_zh) || stringValue(labels['zh-CN']) || undefined,
      description: (typeof parameter.description === 'string' ? parameter.description : stringValue(descriptions['en-US'])) || undefined,
      description_zh: stringValue(parameter.description_zh) || stringValue(descriptions['zh-CN']) || undefined,
      default: parameter.default,
      required: Boolean(parameter.required),
      min: typeof parameter.min === 'number' ? parameter.min : undefined,
      max: typeof parameter.max === 'number' ? parameter.max : undefined,
      options: options.length ? options : undefined,
    }];
  });
}

function normalizeJsonParameterSchema(value: unknown): ParameterSchema[] {
  const schema = isRecord(value) ? value : {};
  const properties = isRecord(schema.properties) ? schema.properties : {};
  const required = new Set(Array.isArray(schema.required) ? schema.required.map(String) : []);
  return Object.entries(properties).flatMap(([key, raw]): ParameterSchema[] => {
    if (!isRecord(raw)) return [];
    const title = isRecord(raw.title) ? raw.title : {};
    const description = isRecord(raw.description) ? raw.description : {};
    const enumValues = Array.isArray(raw.enum)
      ? raw.enum.filter((item): item is string | number => typeof item === 'string' || typeof item === 'number')
      : [];
    const rawType = stringValue(raw.type, 'string');
    const type: ParameterSchema['type'] = enumValues.length
      ? 'select'
      : rawType === 'integer' ? 'integer'
        : rawType === 'number' ? 'number'
          : rawType === 'boolean' ? 'boolean'
            : stringValue(raw.format).startsWith('local-') ? 'path' : 'string';
    return [{
      key,
      type,
      label: (typeof raw.title === 'string' ? raw.title : stringValue(title['en-US'])) || key,
      label_zh: stringValue(raw.title_zh) || stringValue(title['zh-CN']) || undefined,
      description: (typeof raw.description === 'string' ? raw.description : stringValue(description['en-US'])) || undefined,
      description_zh: stringValue(raw.description_zh) || stringValue(description['zh-CN']) || undefined,
      default: raw.default,
      required: required.has(key),
      min: typeof raw.minimum === 'number' ? raw.minimum : undefined,
      max: typeof raw.maximum === 'number' ? raw.maximum : undefined,
      options: enumValues.length ? enumValues.map((item) => ({ label: String(item), value: item })) : undefined,
    }];
  });
}

function localizedFields(row: Record<string, unknown>, fallback: string) {
  const label = isRecord(row.label) ? row.label : isRecord(row.labels) ? row.labels : {};
  const description = isRecord(row.description) ? row.description : {};
  return {
    name: stringValue(row.name) || stringValue(label['en-US']) || fallback,
    name_zh: stringValue(row.name_zh) || stringValue(label['zh-CN']) || undefined,
    description: (typeof row.description === 'string' ? row.description : stringValue(description['en-US'])) || undefined,
    description_zh: stringValue(row.description_zh) || stringValue(description['zh-CN']) || undefined,
  };
}

function normalizeDeploymentCapabilities(value: unknown): DeploymentCapabilities {
  const root = isRecord(value) ? value : {};
  const catalog = isRecord(root.catalog) ? root.catalog : root;
  const components = isRecord(catalog.components) ? catalog.components : {};
  const componentMaps = {
    backbones: new Map(recordArray(components.backbones).map((row) => [stringValue(row.id), row])),
    action_heads: new Map(recordArray(components.action_heads).map((row) => [stringValue(row.id), row])),
  };
  const adapters: DeploymentAdapter[] = recordArray(catalog.adapters ?? catalog.deployment_adapters).flatMap((row) => {
    const id = stringValue(row.id);
    if (!id) return [];
    const protocol = isRecord(row.protocol) ? row.protocol : {};
    const rawParameters = normalizeParameters(row.parameters);
    const parameters = (rawParameters.length ? rawParameters : normalizeJsonParameterSchema(row.parameter_schema))
      .filter((parameter) => !['port', 'idle_timeout_seconds'].includes(parameter.key));
    return [{
      id,
      ...localizedFields(row, id),
      protocol: stringValue(row.protocol) || stringValue(protocol.request_format) || stringValue(protocol.transport) || undefined,
      parameters,
    }];
  });
  const combinations: DeploymentCombination[] = recordArray(catalog.combinations ?? catalog.deployment_combinations).flatMap((row) => {
    const id = stringValue(row.id);
    if (!id) return [];
    const backboneId = stringValue(row.backbone_id) || stringValue(row.backbone);
    const actionHeadId = stringValue(row.action_head_id) || stringValue(row.action_head);
    const backbone = componentMaps.backbones.get(backboneId);
    const actionHead = componentMaps.action_heads.get(actionHeadId);
    const backboneFields = backbone ? localizedFields(backbone, backboneId) : { name: backboneId, name_zh: undefined };
    const actionHeadFields = actionHead ? localizedFields(actionHead, actionHeadId) : { name: actionHeadId, name_zh: undefined };
    const explicit = localizedFields(row, '');
    const rawParameters = normalizeParameters(row.parameters);
    return [{
      id,
      name: explicit.name || `${backboneFields.name} + ${actionHeadFields.name}` || id,
      name_zh: explicit.name_zh || `${backboneFields.name_zh ?? backboneFields.name} + ${actionHeadFields.name_zh ?? actionHeadFields.name}`,
      description: explicit.description,
      description_zh: explicit.description_zh,
      status: row.status === 'experimental' ? 'experimental' : row.status === 'unsupported' ? 'unsupported' : 'verified',
      adapter_id: stringValue(row.adapter_id) || stringValue(row.adapter),
      backbone_id: backboneId,
      action_head_id: actionHeadId,
      recommended_gpu_count: typeof row.recommended_gpu_count === 'number'
        ? row.recommended_gpu_count
        : typeof row.min_gpus === 'number' ? row.min_gpus : undefined,
      parameters: rawParameters.length ? rawParameters : normalizeJsonParameterSchema(row.parameter_schema),
    }];
  });
  return {
    adapters,
    combinations: combinations.filter((item) => item.status !== 'unsupported'),
    experimental_allowed: Boolean(
      root.experimental_allowed
      ?? root.include_experimental
      ?? (isRecord(catalog.filters) ? catalog.filters.include_experimental : false),
    ),
  };
}

function normalizeDeployment(value: unknown): ModelDeployment {
  const row = isRecord(value) ? value : {};
  const endpointValue = row.endpoint;
  const endpointRow = isRecord(endpointValue) ? endpointValue : {};
  const endpointUrl = typeof endpointValue === 'string'
    ? endpointValue
    : stringValue(endpointRow.url) || stringValue(endpointRow.websocket_url);
  const requestedIds = Array.isArray(row.requested_gpu_ids) ? row.requested_gpu_ids.map(Number) : [];
  const assignedIds = Array.isArray(row.assigned_gpu_ids) ? row.assigned_gpu_ids.map(Number) : [];
  return {
    id: stringValue(row.id),
    name: stringValue(row.name) || `Deployment ${stringValue(row.id).slice(0, 8)}`,
    owner_id: stringValue(row.owner_id) || undefined,
    owner_name: stringValue(row.owner_name) || undefined,
    status: stringValue(row.status, 'queued') as DeploymentStatus,
    queue_position: typeof row.queue_position === 'number' ? row.queue_position : undefined,
    checkpoint_id: stringValue(row.checkpoint_id) || undefined,
    checkpoint_path: stringValue(row.checkpoint_path) || undefined,
    combination_id: stringValue(row.combination_id),
    adapter_id: stringValue(row.adapter_id) || undefined,
    backbone_id: stringValue(row.backbone_id) || undefined,
    action_head_id: stringValue(row.action_head_id) || undefined,
    requested_gpu_count: numberValue(row.requested_gpu_count, requestedIds.length || assignedIds.length || 1),
    requested_gpu_ids: requestedIds,
    assigned_gpu_ids: assignedIds,
    endpoint: {
      scope: row.endpoint_scope === 'lan' || endpointRow.scope === 'lan' ? 'lan' : 'local',
      bind_host: stringValue(row.bind_host) || stringValue(endpointRow.bind_host) || undefined,
      advertised_host: stringValue(row.advertised_host) || stringValue(endpointRow.advertised_host) || undefined,
      port: typeof row.port === 'number' ? row.port : typeof endpointRow.port === 'number' ? endpointRow.port : undefined,
      url: endpointUrl || undefined,
    },
    idle_timeout_seconds: numberValue(row.idle_timeout_seconds, 1800),
    parameters: isRecord(row.parameters) ? row.parameters : {},
    log_path: stringValue(row.log_path) || undefined,
    pid: typeof row.pid === 'number' ? row.pid : undefined,
    error_summary: stringValue(row.error) || stringValue(row.error_summary) || undefined,
    queued_at: stringValue(row.queued_at) || undefined,
    started_at: stringValue(row.started_at) || undefined,
    ready_at: stringValue(row.ready_at) || undefined,
    finished_at: stringValue(row.finished_at) || undefined,
    created_at: stringValue(row.created_at) || stringValue(row.queued_at) || undefined,
    managed_inference_available: typeof row.managed_inference_available === 'boolean'
      ? row.managed_inference_available
      : undefined,
  };
}

function normalizeDeploymentMutation(value: unknown): DeploymentMutationResult {
  const row = isRecord(value) ? value : {};
  return {
    deployment: normalizeDeployment(row.deployment ?? row),
    api_key: stringValue(row.api_key) || undefined,
  };
}

function normalizeInferenceRun(value: unknown): InferenceRun {
  const row = isRecord(value) ? value : {};
  return {
    id: stringValue(row.id),
    request_id: stringValue(row.request_id),
    deployment_id: stringValue(row.deployment_id) || undefined,
    owner_id: stringValue(row.owner_id) || undefined,
    status: stringValue(row.status, 'failed') as InferenceRun['status'],
    batch_size: numberValue(row.batch_size, 1),
    image_count: numberValue(row.image_count),
    save_inputs: Boolean(row.save_inputs),
    inputs_saved: Boolean(row.inputs_saved),
    request_summary: isRecord(row.request_summary) ? row.request_summary : {},
    deployment_metadata: isRecord(row.deployment_metadata) ? row.deployment_metadata : {},
    output: isRecord(row.output) ? row.output : undefined,
    latency_ms: typeof row.latency_ms === 'number' ? row.latency_ms : undefined,
    error: stringValue(row.error) || undefined,
    created_at: stringValue(row.created_at) || undefined,
    finished_at: stringValue(row.finished_at) || undefined,
  };
}

function normalizeModelPublication(value: unknown): ModelPublication {
  const row = isRecord(value) ? value : {};
  return {
    id: stringValue(row.id),
    owner_id: stringValue(row.owner_id),
    checkpoint_id: stringValue(row.checkpoint_id) || undefined,
    utility_run_id: stringValue(row.utility_run_id) || undefined,
    provider: 'huggingface',
    repo_id: stringValue(row.repo_id),
    revision: stringValue(row.revision, 'main'),
    private: row.private !== false,
    status: stringValue(row.status, 'queued'),
    source_path: stringValue(row.source_path),
    result_url: stringValue(row.result_url) || undefined,
    metadata: isRecord(row.metadata) ? row.metadata : {},
    error: stringValue(row.error) || undefined,
    created_at: stringValue(row.created_at) || undefined,
    started_at: stringValue(row.started_at) || undefined,
    finished_at: stringValue(row.finished_at) || undefined,
    updated_at: stringValue(row.updated_at) || undefined,
  };
}

function evaluationLocalized(row: Record<string, unknown>, fallback: string) {
  const title = isRecord(row.title) ? row.title : isRecord(row.label) ? row.label : {};
  const description = isRecord(row.description) ? row.description : {};
  return {
    name: (typeof row.title === 'string' ? row.title : stringValue(row.name) || stringValue(title['en-US'])) || fallback,
    name_zh: stringValue(row.name_zh) || stringValue(title['zh-CN']) || undefined,
    description: (typeof row.description === 'string' ? row.description : stringValue(description['en-US'])) || undefined,
    description_zh: stringValue(row.description_zh) || stringValue(description['zh-CN']) || undefined,
  };
}

function normalizeEvaluationChoices(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.flatMap((raw) => {
    if (typeof raw === 'string' || typeof raw === 'number') {
      const id = String(raw);
      return [{ value: id, label: id }];
    }
    if (!isRecord(raw)) return [];
    const id = stringValue(raw.value) || stringValue(raw.id) || stringValue(raw.name);
    if (!id) return [];
    const fields = evaluationLocalized(raw, id);
    return [{ value: id, label: fields.name, label_zh: fields.name_zh, description: fields.description, description_zh: fields.description_zh }];
  });
}

function normalizeEvaluationPresets(value: unknown, defaults: Record<string, unknown>) {
  const rows: Record<string, unknown>[] = Array.isArray(value)
    ? value.filter(isRecord)
    : isRecord(value) ? Object.entries(value).map(([id, raw]): Record<string, unknown> => isRecord(raw) ? { id, ...raw } : { id }) : [];
  const valid = new Set<EvaluationPreset>(['quick', 'standard', 'full', 'custom']);
  const presets = rows.flatMap((row) => {
    const id = stringValue(row.id) as EvaluationPreset;
    if (!valid.has(id)) return [];
    const fields = evaluationLocalized(row, id);
    const parameters = isRecord(row.parameters)
      ? row.parameters
      : isRecord(row.defaults) ? row.defaults
        : Object.fromEntries(Object.entries(row).filter(([key]) => !['id', 'label', 'title', 'description'].includes(key)));
    return [{ id, ...fields, parameters }];
  });
  if (presets.length) return presets;
  return (['quick', 'standard', 'full', 'custom'] as EvaluationPreset[]).map((id) => ({
    id,
    name: id,
    parameters: isRecord(defaults[id]) ? defaults[id] : {},
  }));
}

function normalizeEvaluationEnvironment(value: unknown) {
  const rows: Record<string, unknown>[] = Array.isArray(value)
    ? value.filter(isRecord)
    : isRecord(value)
      ? (Array.isArray(value.checks) ? value.checks.filter(isRecord) : Object.entries(value).map(([id, raw]): Record<string, unknown> => isRecord(raw) ? { id, ...raw } : { id, configured: Boolean(raw) }))
      : [];
  return rows.map((row) => {
    const fields = evaluationLocalized(row, stringValue(row.id) || stringValue(row.key));
    const configured = row.available ?? row.ready ?? row.ok ?? row.configured;
    const message = isRecord(row.message) ? row.message : {};
    return {
      id: stringValue(row.id) || stringValue(row.key) || stringValue(row.code),
      configured: configured !== false,
      required: row.required !== false,
      title: fields.name || undefined,
      title_zh: fields.name_zh,
      message: (typeof row.message === 'string' ? row.message : stringValue(message['en-US'])) || fields.description,
      message_zh: stringValue(message['zh-CN']) || fields.description_zh,
    };
  });
}

function normalizeEvaluationBenchmark(value: unknown): EvaluationBenchmark {
  const row = isRecord(value) ? value : {};
  const id = stringValue(row.id);
  const fields = evaluationLocalized(row, id);
  const defaults = {
    ...(isRecord(row.defaults) ? row.defaults : {}),
    ...(row.default_suite != null ? { suite: row.default_suite } : {}),
    ...(row.default_task_set != null ? { task_set: row.default_task_set } : {}),
    ...(row.default_split != null ? { split: row.default_split } : {}),
  };
  const compatibility = isRecord(row.compatibility) ? row.compatibility : {};
  const combinationValues = compatibility.combination_ids
    ?? compatibility.combinations
    ?? row.combination_ids
    ?? row.compatible_combinations;
  const backboneValues = compatibility.backbone_ids ?? compatibility.backbones;
  const actionHeadValues = compatibility.action_head_ids ?? compatibility.action_heads;
  const environment = normalizeEvaluationEnvironment(row.environment);
  const readiness = isRecord(row.readiness) ? row.readiness : {};
  const status = row.status === 'experimental' ? 'experimental' : row.status === 'unsupported' ? 'unsupported' : 'verified';
  return {
    id,
    ...fields,
    status,
    available: row.available !== false && row.enabled !== false && readiness.ready !== false && status !== 'unsupported' && !environment.some((item) => item.required && !item.configured),
    suites: normalizeEvaluationChoices(row.suites),
    task_sets: normalizeEvaluationChoices(row.task_sets ?? row.taskSets),
    splits: normalizeEvaluationChoices(row.splits),
    presets: normalizeEvaluationPresets(row.presets, defaults),
    defaults,
    parameters: normalizeJsonParameterSchema(row.parameter_schema ?? row.parameters),
    compatibility: {
      combination_ids: Array.isArray(combinationValues) ? combinationValues.map((item) => isRecord(item) ? stringValue(item.id) : String(item)).filter(Boolean) : [],
      backbone_ids: Array.isArray(backboneValues) ? backboneValues.map(String) : undefined,
      action_head_ids: Array.isArray(actionHeadValues) ? actionHeadValues.map(String) : undefined,
    },
    environment,
  };
}

function normalizeEvaluationCapabilities(value: unknown): EvaluationCapabilities {
  const root = isRecord(value) ? value : {};
  const catalog = isRecord(root.catalog) ? root.catalog : root;
  const rawBenchmarks = Array.isArray(value)
    ? value
    : catalog.benchmarks ?? root.items ?? root.data;
  const globalPresets = catalog.presets ?? root.presets;
  return {
    benchmarks: (Array.isArray(rawBenchmarks) ? rawBenchmarks : [])
      .map((item) => isRecord(item) && item.presets == null && Array.isArray(globalPresets) ? { ...item, presets: globalPresets } : item)
      .map(normalizeEvaluationBenchmark)
      .filter((item) => item.id && item.status !== 'unsupported'),
    experimental_allowed: Boolean(root.experimental_allowed ?? root.include_experimental),
  };
}

function normalizeEvaluationWandb(value: unknown): EvaluationWandbConfig {
  const row = isRecord(value) ? value : {};
  const allowed = new Set(['summary', 'tasks', 'config', 'videos']);
  return {
    enabled: Boolean(row.enabled),
    mode: row.mode === 'offline' ? 'offline' : 'online',
    project: stringValue(row.project, 'AlphaBrain-Evaluation'),
    entity: stringValue(row.entity),
    run_name: stringValue(row.run_name),
    tags: Array.isArray(row.tags) ? row.tags.map(String) : [],
    categories: Array.isArray(row.categories)
      ? row.categories.filter((item): item is EvaluationWandbConfig['categories'][number] => typeof item === 'string' && allowed.has(item))
      : ['summary', 'tasks', 'config'],
  };
}

function rateValue(value: unknown, succeeded: number, total: number): number {
  const explicit = numberValue(value, Number.NaN);
  if (Number.isFinite(explicit)) return explicit > 1 ? explicit / 100 : explicit;
  return total ? succeeded / total : 0;
}

function normalizeEvaluationVideo(value: unknown, evaluationId?: string): EvaluationVideo {
  const row = isRecord(value) ? value : {};
  const artifactId = stringValue(row.id) || stringValue(row.artifact_id);
  const id = artifactId || stringValue(row.path);
  return {
    id,
    name: stringValue(row.name) || stringValue(row.filename) || stringValue(row.task_name) || id.split('/').pop() || 'video',
    task_id: stringValue(row.task_id) || stringValue(row.task) || undefined,
    task_name: stringValue(row.task_name) || undefined,
    episode_index: typeof row.episode_index === 'number' ? row.episode_index : typeof row.episode === 'number' ? row.episode : undefined,
    success: typeof row.success === 'boolean' ? row.success : undefined,
    path: stringValue(row.path) || undefined,
    url: stringValue(row.url) || stringValue(row.download_url) || (evaluationId && artifactId ? `${API_BASE}/evaluations/${encodeURIComponent(evaluationId)}/artifacts/${encodeURIComponent(artifactId)}` : undefined),
    size_bytes: typeof row.size_bytes === 'number' ? row.size_bytes : undefined,
  };
}

function normalizeEvaluationArtifact(value: unknown, evaluationId?: string): EvaluationArtifact {
  const row = isRecord(value) ? value : {};
  const id = stringValue(row.id) || stringValue(row.artifact_id) || undefined;
  const path = stringValue(row.path);
  return {
    id,
    kind: stringValue(row.kind, 'file'),
    path,
    name: stringValue(row.name) || stringValue(row.filename) || path.split('/').pop() || id || 'artifact',
    media_type: stringValue(row.media_type) || stringValue(row.content_type) || undefined,
    url: stringValue(row.url) || stringValue(row.download_url)
      || (evaluationId && id ? `${API_BASE}/evaluations/${encodeURIComponent(evaluationId)}/artifacts/${encodeURIComponent(id)}` : undefined),
    size_bytes: typeof row.size_bytes === 'number' ? row.size_bytes : undefined,
    metadata: isRecord(row.metadata) ? row.metadata : {},
  };
}

function normalizeEvaluationMatrix(value: unknown): EvaluationMatrix {
  const row = isRecord(value) ? value : {};
  return {
    id: stringValue(row.id) || stringValue(row.name) || 'matrix',
    name: stringValue(row.name) || stringValue(row.id) || 'Matrix',
    row_labels: Array.isArray(row.row_labels) ? row.row_labels.map(String) : [],
    column_labels: Array.isArray(row.column_labels) ? row.column_labels.map(String) : [],
    values: Array.isArray(row.values)
      ? row.values.map((raw) => Array.isArray(raw) ? raw.map((item) => typeof item === 'number' ? item : null) : [])
      : [],
    value_format: stringValue(row.value_format, 'rate'),
    metadata: isRecord(row.metadata) ? row.metadata : {},
  };
}

function normalizeEvaluationSeries(value: unknown): EvaluationSeries {
  const row = isRecord(value) ? value : {};
  return {
    id: stringValue(row.id) || stringValue(row.name) || 'series',
    name: stringValue(row.name) || stringValue(row.id) || 'Series',
    x_label: stringValue(row.x_label),
    y_label: stringValue(row.y_label),
    points: recordArray(row.points).map((point) => ({
      x: typeof point.x === 'number' ? point.x : stringValue(point.x),
      y: typeof point.y === 'number' ? point.y : null,
      metadata: isRecord(point.metadata) ? point.metadata : {},
    })),
    metadata: isRecord(row.metadata) ? row.metadata : {},
  };
}

function normalizeEvaluationResult(value: unknown, evaluationId?: string): EvaluationResult {
  const root = isRecord(value) ? value : {};
  const row = isRecord(root.result) ? root.result : root;
  const summary = isRecord(row.summary) ? row.summary : {};
  const succeeded = numberValue(summary.num_successes ?? summary.succeeded ?? summary.successes ?? row.num_successes ?? row.succeeded ?? row.successes);
  const total = numberValue(summary.num_episodes ?? summary.total ?? summary.episodes ?? row.num_episodes ?? row.total);
  const tasks = recordArray(row.tasks).map((task) => {
    const taskSucceeded = numberValue(task.succeeded ?? task.successes ?? task.num_successes);
    const taskTotal = numberValue(task.total ?? task.episodes ?? task.num_episodes);
    return {
      id: stringValue(task.id) || stringValue(task.task_id) || stringValue(task.name),
      name: stringValue(task.name) || undefined,
      suite: stringValue(task.suite) || undefined,
      num_successes: taskSucceeded,
      num_episodes: taskTotal,
      success_rate: rateValue(task.success_rate, taskSucceeded, taskTotal),
      duration_seconds: typeof task.duration_seconds === 'number' ? task.duration_seconds : undefined,
      metadata: isRecord(task.metadata) ? task.metadata : {},
    };
  });
  return {
    schema_version: stringValue(row.schema_version, 'evaluation-result-v1'),
    kind: stringValue(row.kind) || undefined,
    status: stringValue(row.status) || undefined,
    evaluation_id: stringValue(row.evaluation_id) || evaluationId,
    benchmark_id: stringValue(row.benchmark_id) || stringValue(row.benchmark) || undefined,
    summary: {
      ...summary,
      success_rate: rateValue(summary.success_rate ?? row.success_rate, succeeded, total),
      num_tasks: typeof summary.num_tasks === 'number' ? summary.num_tasks : tasks.length,
      num_successes: succeeded,
      num_episodes: total,
    },
    duration_seconds: typeof row.duration_seconds === 'number'
      ? row.duration_seconds
      : typeof summary.duration_seconds === 'number' ? summary.duration_seconds : undefined,
    tasks,
    episodes: recordArray(row.episodes).map((episode) => ({
      id: stringValue(episode.id) || `${stringValue(episode.task_id)}-${numberValue(episode.episode ?? episode.episode_index)}`,
      task_id: stringValue(episode.task_id) || stringValue(episode.task) || undefined,
      task_name: stringValue(episode.task_name) || undefined,
      episode_index: typeof episode.episode_index === 'number' ? episode.episode_index : typeof episode.episode === 'number' ? episode.episode : undefined,
      seed: typeof episode.seed === 'number' ? episode.seed : undefined,
      success: Boolean(episode.success ?? episode.succeeded),
      duration_seconds: typeof episode.duration_seconds === 'number' ? episode.duration_seconds : undefined,
      steps: typeof episode.steps === 'number' ? episode.steps : undefined,
      video_id: stringValue(episode.video_id) || stringValue(episode.video) || undefined,
    })),
    videos: recordArray(row.videos).map((video) => normalizeEvaluationVideo(video, evaluationId)),
    artifacts: recordArray(row.artifacts).map((artifact) => normalizeEvaluationArtifact(artifact, evaluationId)),
    matrices: recordArray(row.matrices).map(normalizeEvaluationMatrix),
    series: recordArray(row.series).map(normalizeEvaluationSeries),
    comparisons: recordArray(row.comparisons),
    children: recordArray(row.children).map((child) => ({
      evaluation_id: stringValue(child.evaluation_id) || stringValue(child.id),
      name: stringValue(child.name),
      position: numberValue(child.position),
      kind: stringValue(child.kind, 'standard'),
      status: stringValue(child.status, 'queued'),
      checkpoint: stringValue(child.checkpoint) || undefined,
      suite: stringValue(child.suite) || undefined,
      result_path: stringValue(child.result_path) || undefined,
      summary: isRecord(child.summary) ? child.summary : {},
      error: stringValue(child.error) || undefined,
    })),
    parameters: isRecord(row.parameters) ? row.parameters : {},
    metadata: isRecord(row.metadata) ? row.metadata : {},
    error: stringValue(row.error) || undefined,
  };
}

function normalizeEvaluation(value: unknown): EvaluationRun {
  const row = isRecord(value) ? value : {};
  const id = stringValue(row.id);
  const rawStatus = stringValue(row.status, 'queued');
  const status = (rawStatus === 'succeeded' ? 'completed' : rawStatus) as EvaluationStatus;
  const requestedIds = Array.isArray(row.requested_gpu_ids) ? row.requested_gpu_ids.map(Number) : [];
  const assignedIds = Array.isArray(row.assigned_gpu_ids) ? row.assigned_gpu_ids.map(Number) : [];
  const rawResult = isRecord(row.result)
    ? row.result
    : isRecord(row.result_summary) && Object.keys(row.result_summary).length > 0
      ? { summary: row.result_summary, benchmark_id: row.benchmark_id }
      : isRecord(row.metrics) ? row.metrics : undefined;
  const result = rawResult ? normalizeEvaluationResult(rawResult, id) : undefined;
  const videos = recordArray(row.videos).map((video) => normalizeEvaluationVideo(video, id));
  return {
    id,
    group_id: stringValue(row.group_id) || undefined,
    position: typeof row.position === 'number' ? row.position : undefined,
    evaluation_kind: stringValue(row.evaluation_kind, 'standard'),
    source_kind: stringValue(row.source_kind) || undefined,
    deployment_id: stringValue(row.deployment_id) || undefined,
    result_schema_version: stringValue(row.result_schema_version) || undefined,
    name: stringValue(row.name) || `Evaluation ${id.slice(0, 8)}`,
    owner_id: stringValue(row.owner_id) || undefined,
    owner_name: stringValue(row.owner_name) || undefined,
    status,
    queue_position: typeof row.queue_position === 'number' ? row.queue_position : undefined,
    checkpoint_id: stringValue(row.checkpoint_id) || undefined,
    checkpoint_path: stringValue(row.checkpoint_path) || undefined,
    combination_id: stringValue(row.combination_id) || stringValue(row.combination),
    adapter_id: stringValue(row.adapter_id) || undefined,
    backbone_id: stringValue(row.backbone_id) || undefined,
    action_head_id: stringValue(row.action_head_id) || undefined,
    benchmark_id: stringValue(row.benchmark_id) || (isRecord(row.benchmark) ? stringValue(row.benchmark.id) : stringValue(row.benchmark)),
    preset: (['quick', 'standard', 'full', 'custom'].includes(stringValue(row.preset)) ? stringValue(row.preset) : 'quick') as EvaluationPreset,
    suite: stringValue(row.suite) || (isRecord(row.suite) ? stringValue(row.suite.id) || stringValue(row.suite.name) : undefined),
    task_set: stringValue(row.task_set) || undefined,
    split: stringValue(row.split) || undefined,
    parameters: isRecord(row.parameters) ? row.parameters : {},
    wandb: normalizeEvaluationWandb(row.wandb),
    wandb_status: stringValue(row.wandb_status) || undefined,
    wandb_error: stringValue(row.wandb_error) || undefined,
    wandb_run_url: stringValue(row.wandb_run_url) || undefined,
    requested_gpu_ids: requestedIds,
    assigned_gpu_ids: assignedIds,
    requested_gpu_count: typeof row.requested_gpu_count === 'number' ? row.requested_gpu_count : undefined,
    progress: typeof row.progress === 'number' ? row.progress : undefined,
    phase: stringValue(row.phase) || undefined,
    output_path: stringValue(row.output_path) || stringValue(row.output_dir) || undefined,
    log_path: stringValue(row.log_path) || undefined,
    result_path: stringValue(row.result_path) || undefined,
    pid: typeof row.pid === 'number' ? row.pid : undefined,
    exit_code: typeof row.exit_code === 'number' ? row.exit_code : undefined,
    error_summary: stringValue(row.error) || stringValue(row.error_summary) || undefined,
    result,
    videos: videos.length ? videos : result?.videos ?? [],
    queued_at: stringValue(row.queued_at) || undefined,
    started_at: stringValue(row.started_at) || undefined,
    finished_at: stringValue(row.finished_at) || undefined,
    created_at: stringValue(row.created_at) || stringValue(row.queued_at) || undefined,
  };
}

function normalizeEvaluationGroup(value: unknown): EvaluationGroup {
  const row = isRecord(value) ? value : {};
  const id = stringValue(row.id);
  return {
    id,
    owner_id: stringValue(row.owner_id) || undefined,
    owner_name: stringValue(row.owner_name) || undefined,
    name: stringValue(row.name) || `Evaluation group ${id.slice(0, 8)}`,
    kind: stringValue(row.kind, 'batch'),
    status: stringValue(row.status, 'queued'),
    spec: isRecord(row.spec) ? row.spec : {},
    result_summary: isRecord(row.result_summary) ? row.result_summary : {},
    result_path: stringValue(row.result_path) || undefined,
    error: stringValue(row.error) || undefined,
    runs: recordArray(row.runs).map(normalizeEvaluation),
    result: isRecord(row.result) ? normalizeEvaluationResult(row.result) : undefined,
    created_at: stringValue(row.created_at) || undefined,
    updated_at: stringValue(row.updated_at) || undefined,
    started_at: stringValue(row.started_at) || undefined,
    finished_at: stringValue(row.finished_at) || undefined,
  };
}

function normalizeEvaluationComparison(value: unknown): EvaluationComparison {
  const row = isRecord(value) ? value : {};
  const items = recordArray(row.items ?? row.evaluations).map((item) => {
    const evaluation = normalizeEvaluation(item.evaluation ?? item);
    return {
      evaluation,
      result: isRecord(item.result) ? normalizeEvaluationResult(item.result, evaluation.id) : undefined,
    };
  });
  const signature = isRecord(row.signature) ? row.signature : undefined;
  return {
    benchmark_id: stringValue(row.benchmark_id) || items[0]?.evaluation.benchmark_id,
    signature: signature ? {
      suite: stringValue(signature.suite) || undefined,
      task_set: stringValue(signature.task_set) || undefined,
      split: stringValue(signature.split) || undefined,
    } : stringValue(row.signature) || undefined,
    compatible: row.compatible !== false,
    warnings: Array.isArray(row.warnings) ? row.warnings.map(String) : [],
    items,
  };
}

function normalizeEvaluationCheckpointInspection(value: unknown): EvaluationCheckpointInspection {
  const root = isRecord(value) ? value : {};
  const row = isRecord(root.inspection) ? root.inspection : root;
  const detected = isRecord(row.detected) ? row.detected : {};
  const evaluationCandidates = recordArray(row.evaluation_candidates ?? row.evaluationCandidates ?? row.candidates).flatMap((candidate) => {
    const combinationId = stringValue(candidate.combination_id) || stringValue(candidate.id);
    if (!combinationId) return [];
    return [{
      combination_id: combinationId,
      benchmark_ids: Array.isArray(candidate.benchmark_ids)
        ? candidate.benchmark_ids.map(String).filter(Boolean)
        : [],
    }];
  });
  return {
    valid: typeof row.valid === 'boolean' ? row.valid : typeof row.ok === 'boolean' ? row.ok : undefined,
    detected: {
      framework: stringValue(detected.framework) || undefined,
      backbone: stringValue(detected.backbone) || undefined,
      action_head: stringValue(detected.action_head) || undefined,
      adapter_id: stringValue(detected.adapter_id) || undefined,
      combination_id: stringValue(detected.combination_id) || undefined,
      candidate_combination_ids: Array.isArray(detected.candidate_combination_ids)
        ? detected.candidate_combination_ids.map(String).filter(Boolean)
        : evaluationCandidates.map((candidate) => candidate.combination_id),
    },
    evaluation_candidates: evaluationCandidates,
    compatible_benchmark_ids: Array.isArray(row.compatible_benchmark_ids)
      ? row.compatible_benchmark_ids.map(String).filter(Boolean)
      : [],
    issues: normalizeIssues(row.issues),
  };
}

function localizedWandbText(value: unknown): Partial<Record<'zh-CN' | 'en-US', string>> {
  const row = isRecord(value) ? value : {};
  return {
    ...(stringValue(row['zh-CN']) ? { 'zh-CN': stringValue(row['zh-CN']) } : {}),
    ...(stringValue(row['en-US']) ? { 'en-US': stringValue(row['en-US']) } : {}),
  };
}

function normalizeWorkflowSchema(value: unknown): WorkflowSchema {
  const row = isRecord(value) ? value : {};
  return {
    id: stringValue(row.id, 'standard'),
    schema_version: numberValue(row.schema_version, 1),
    title: localizedWandbText(row.title),
    description: localizedWandbText(row.description),
    fields: recordArray(row.fields).map((field) => ({
      key: stringValue(field.key),
      type: stringValue(field.type, 'string') as WorkflowSchema['fields'][number]['type'],
      label: localizedWandbText(field.label),
      description: localizedWandbText(field.description),
      default: field.default,
      required: Boolean(field.required),
      min: typeof field.min === 'number' ? field.min : undefined,
      max: typeof field.max === 'number' ? field.max : undefined,
      options: recordArray(field.options).map((option) => ({
        value: typeof option.value === 'number' ? option.value : stringValue(option.value),
        label: localizedWandbText(option.label),
      })),
      visible_when: isRecord(field.visible_when) ? {
        key: stringValue(field.visible_when.key),
        ...('equals' in field.visible_when ? { equals: field.visible_when.equals } : {}),
        ...(Array.isArray(field.visible_when.in) ? { in: field.visible_when.in } : {}),
      } : undefined,
      methods: Array.isArray(field.methods) ? field.methods.map(String) : undefined,
      algorithms: Array.isArray(field.algorithms) ? field.algorithms.map(String) : undefined,
      config_path: stringValue(field.config_path) || undefined,
      stage: stringValue(field.stage) || undefined,
    })),
  };
}

function normalizeWandbCategoryCapabilities(value: unknown): WandbCategoryCapability[] {
  return recordArray(value).flatMap((row) => {
    const id = stringValue(row.id) as WandbUploadCategory;
    if (!WANDB_CATEGORY_IDS.has(id)) return [];
    return [{
      id,
      supported: Boolean(row.supported),
      title: localizedWandbText(row.title),
      description: localizedWandbText(row.description),
      ...(isRecord(row.reason) ? { reason: localizedWandbText(row.reason) } : {}),
    }];
  });
}

function normalizeCapabilities(value: unknown): CapabilityResponse {
  const root = isRecord(value) ? value : {};
  const catalog = isRecord(root.catalog) ? root.catalog : root;
  const components = isRecord(catalog.components) ? catalog.components : {};
  const combinations = recordArray(catalog.combinations).map((combo) => ({
    id: stringValue(combo.id),
    status: combo.status === 'experimental' ? 'experimental' as const : combo.status === 'unsupported' ? 'unsupported' as const : 'verified' as const,
    backbone: stringValue(combo.backbone),
    action_head: stringValue(combo.action_head),
    method: stringValue(combo.method),
    datasets: Array.isArray(combo.datasets) ? combo.datasets.map(String) : [],
    min_gpus: typeof combo.min_gpus === 'number' ? combo.min_gpus : undefined,
    launcher: stringValue(combo.launcher) || undefined,
    rl_algorithm: stringValue(combo.rl_algorithm) || undefined,
    workflow: normalizeWorkflowSchema(combo.workflow),
    wandb_categories: normalizeWandbCategoryCapabilities(combo.wandb_categories),
  }));
  const make = (enrichedGroup: string, componentGroup: string, kind: CapabilityKind) => {
    const enriched = recordArray(catalog[enrichedGroup]);
    const source = enriched.length ? enriched : recordArray(components[componentGroup]);
    return source.map((item) => capabilityFrom(item, kind, combinations));
  };
  return {
    backbones: make('backbones', 'backbones', 'backbone'),
    action_heads: make('action_heads', 'action_heads', 'action_head'),
    methods: make('methods', 'training_methods', 'method'),
    datasets: make('datasets', 'datasets', 'dataset'),
    combinations,
    experimental_allowed: Boolean(root.include_experimental),
  };
}

function normalizeIssues(value: unknown): PreflightItem[] {
  return recordArray(value).map((row) => {
    const detail = isRecord(row.detail) ? row.detail : {};
    const messages = isRecord(row.message_i18n)
      ? row.message_i18n
      : isRecord(detail.message_i18n) ? detail.message_i18n : {};
    return {
      id: stringValue(row.code) || stringValue(row.id) || undefined,
      level: row.level === 'error' || row.level === 'warning' || row.level === 'success'
        ? row.level
        : row.severity === 'error' || row.severity === 'warning' ? row.severity : 'info',
      title: stringValue(row.title) || stringValue(row.code) || stringValue(row.message),
      message: stringValue(row.message) || undefined,
      message_i18n: {
        ...(stringValue(messages['zh-CN']) ? { 'zh-CN': stringValue(messages['zh-CN']) } : {}),
        ...(stringValue(messages['en-US']) ? { 'en-US': stringValue(messages['en-US']) } : {}),
      },
      path: stringValue(row.field) || stringValue(row.path) || undefined,
      detail,
    };
  });
}

function commandText(value: unknown): string | undefined {
  const stages = recordArray(value);
  if (!stages.length) return undefined;
  return stages.map((stage) => {
    const env = isRecord(stage.environment)
      ? Object.entries(stage.environment).map(([key, item]) => `${key}=${JSON.stringify(String(item))}`).join(' ')
      : '';
    const command = Array.isArray(stage.command) ? stage.command.map((item) => JSON.stringify(String(item))).join(' ') : '';
    const label = stringValue(stage.name) || stringValue(stage.phase);
    return `${label ? `# ${label}\n` : ''}${env ? `${env} ` : ''}${command}`.trim();
  }).join('\n\n');
}

function normalizeResolve(value: unknown): ResolveResult {
  const row = isRecord(value) ? value : {};
  const resolved = isRecord(row.resolved) ? row.resolved : {};
  const issues = normalizeIssues(row.issues);
  return {
    resolved_config: resolved,
    command: commandText(row.command_preview),
    diff: isRecord(row.diff)
      ? Object.entries(row.diff).map(([path, after]) => ({ path, after }))
      : [],
    warnings: issues.filter((item) => item.level === 'warning').map((item) => item.message ?? item.title),
  };
}

function normalizeSettings(value: unknown): SystemSettings {
  const row = isRecord(value) ? value : {};
  const roots = Array.isArray(row.results_roots) ? row.results_roots.map(String) : [];
  const environment = isRecord(row.environment)
    ? Object.fromEntries(Object.entries(row.environment).filter(([key]) => key.toUpperCase() !== 'WANDB_API_KEY'))
    : {};
  return {
    mode: row.deployment_mode === 'lab' ? 'laboratory' : 'personal',
    experimental_allowed: Boolean(row.experimental_globally_enabled),
    results_roots: roots.length ? roots : ['results'],
    results_root: roots[0] ?? 'results',
    storage_monitor_path: stringValue(row.storage_monitor_path) || undefined,
    dataset_roots: Array.isArray(row.dataset_roots) ? row.dataset_roots.map(String) : ['data'],
    managed_dataset_root: stringValue(row.managed_dataset_root) || undefined,
    pretrained_root: stringValue(row.pretrained_root) || undefined,
    cpu_utility_concurrency: numberValue(row.cpu_utility_concurrency, 2),
    model_server_python: stringValue(row.model_server_python)
      || stringValue(environment.ALPHABRAIN_PYTHON)
      || undefined,
    remote_training_enabled: Boolean(row.remote_training_enabled),
    remote_training_host: stringValue(row.remote_training_host) || undefined,
    remote_training_user: stringValue(row.remote_training_user) || undefined,
    remote_training_port: numberValue(row.remote_training_port, 22),
    remote_training_repo_root: stringValue(row.remote_training_repo_root) || undefined,
    remote_training_identity_file: stringValue(row.remote_training_identity_file) || undefined,
    remote_training_gpu_ids: Array.isArray(row.remote_training_gpu_ids)
      ? row.remote_training_gpu_ids.map((value) => Number(value)).filter(Number.isInteger)
      : [],
    remote_training_setup_command: stringValue(row.remote_training_setup_command) || undefined,
    secure_cookies: Boolean(row.secure_cookies),
    secure_cookies_locked: Boolean(row.secure_cookies_locked),
    low_disk_percent: numberValue(row.disk_min_free_percent, 10),
    low_disk_gib: numberValue(row.disk_min_free_gib, 100),
    environment,
  };
}

function settingsPayload(value: Partial<SystemSettings>): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  if (value.mode) payload.deployment_mode = value.mode === 'laboratory' ? 'lab' : 'personal';
  if (value.admin_password) payload.admin_password = value.admin_password;
  if (value.experimental_allowed != null) payload.experimental_globally_enabled = value.experimental_allowed;
  if (value.results_roots !== undefined) payload.results_roots = value.results_roots;
  else if (value.results_root) payload.results_roots = [value.results_root];
  if (value.storage_monitor_path != null) payload.storage_monitor_path = value.storage_monitor_path;
  if (value.dataset_roots) payload.dataset_roots = value.dataset_roots;
  if (value.managed_dataset_root != null) payload.managed_dataset_root = value.managed_dataset_root;
  if (value.pretrained_root != null) payload.pretrained_root = value.pretrained_root;
  if (value.cpu_utility_concurrency != null) payload.cpu_utility_concurrency = value.cpu_utility_concurrency;
  if (value.model_server_python != null) payload.model_server_python = value.model_server_python;
  if (value.remote_training_enabled != null) payload.remote_training_enabled = value.remote_training_enabled;
  if (value.remote_training_host != null) payload.remote_training_host = value.remote_training_host;
  if (value.remote_training_user != null) payload.remote_training_user = value.remote_training_user;
  if (value.remote_training_port != null) payload.remote_training_port = value.remote_training_port;
  if (value.remote_training_repo_root != null) payload.remote_training_repo_root = value.remote_training_repo_root;
  if (value.remote_training_identity_file != null) payload.remote_training_identity_file = value.remote_training_identity_file;
  if (value.remote_training_gpu_ids != null) payload.remote_training_gpu_ids = value.remote_training_gpu_ids.map(Number);
  if (value.remote_training_setup_command != null) payload.remote_training_setup_command = value.remote_training_setup_command;
  if (value.secure_cookies != null) payload.secure_cookies = value.secure_cookies;
  if (value.low_disk_percent != null) payload.disk_min_free_percent = value.low_disk_percent;
  if (value.low_disk_gib != null) payload.disk_min_free_gib = value.low_disk_gib;
  if (isRecord(value.environment)) {
    payload.environment = Object.fromEntries(
      Object.entries(value.environment).filter(([key]) => key.toUpperCase() !== 'WANDB_API_KEY'),
    );
  }
  return payload;
}

export const api = {
  runtime: async (): Promise<{ demo_mode: boolean; version: string }> => {
    const row = await request<Record<string, unknown>>('/health');
    return {
      demo_mode: Boolean(row.demo_mode),
      version: stringValue(row.version),
    };
  },
  setup: {
    status: async (): Promise<SetupStatus> => {
      const row = await request<Record<string, unknown>>('/setup/status');
      return {
        configured: Boolean(row.initialized),
        mode: row.deployment_mode === 'lab' ? 'laboratory' : 'personal',
      };
    },
    create: async (payload: Record<string, unknown>) => {
      const raw = await request<Record<string, unknown>>('/setup', {
        method: 'POST',
        body: body({
          username: payload.username,
          display_name: payload.display_name,
          password: payload.password ?? '',
          mode: payload.mode === 'laboratory' ? 'lab' : 'personal',
        }),
      });
      const mode = payload.mode === 'laboratory' ? 'laboratory' : 'personal';
      return {
        user: { ...normalizeUser(raw.user), deployment_mode: mode },
        status: { configured: true, mode } as SetupStatus,
      };
    },
  },
  auth: {
    login: async (username: string, password: string) => {
      const raw = await request<Record<string, unknown>>('/auth/login', { method: 'POST', body: body({ username, password }) });
      return { user: normalizeUser(raw.user) };
    },
    logout: () => request<void>('/auth/logout', { method: 'POST' }),
    me: async (): Promise<User> => {
      const raw = await request<Record<string, unknown>>('/auth/me');
      return normalizeUser(raw.user ?? raw, Boolean(raw.experimental_available), raw.deployment_mode);
    },
  },
  dashboard: async (): Promise<DashboardData> => {
    const [raw, storage] = await Promise.all([
      request<Record<string, unknown>>('/dashboard'),
      request<unknown[]>('/storage').catch(() => []),
    ]);
    const counts = isRecord(raw.job_counts) ? raw.job_counts : {};
    const systemMetrics = normalizeSystemMetrics(raw.system_metrics ?? raw.system);
    return {
      running_jobs: numberValue(counts.running),
      queued_jobs: numberValue(counts.queued) + numberValue(counts.blocked),
      failed_jobs: numberValue(counts.failed),
      completed_jobs: numberValue(counts.completed),
      recent_experiments: recordArray(raw.recent_experiments).map(normalizeExperiment),
      alerts: recordArray(raw.alerts).map((alert) => ({
        id: stringValue(alert.code) || stringValue(alert.id) || undefined,
        level: alert.level === 'error' || alert.level === 'warning' ? alert.level : 'info',
        title: stringValue(alert.title) || stringValue(alert.code) || 'system_alert',
        message: stringValue(alert.message) || stringValue(alert.path) || undefined,
      })),
      storage: normalizeStorage(storage[0]),
      system_metrics: systemMetrics,
      system: systemMetrics,
      remote_training: isRecord(raw.remote_training) ? {
        enabled: Boolean(raw.remote_training.enabled),
        target: stringValue(raw.remote_training.target) || undefined,
      } : { enabled: false },
    };
  },
  capabilities: async (includeExperimental = false) =>
    normalizeCapabilities(await request<unknown>(`/capabilities?include_experimental=${includeExperimental}`)),
  gpus: async (): Promise<GPUInventory> => normalizeGPUInventory(await request<unknown>('/gpus')),
  trainingTarget: async (): Promise<TrainingTarget> => {
    const raw = await request<unknown>('/training/target');
    const row = isRecord(raw) ? raw : {};
    return {
      mode: row.mode === 'remote' ? 'remote' : 'local',
      gpu_ids: Array.isArray(row.gpu_ids)
        ? Array.from(new Set(row.gpu_ids.filter((value): value is number => Number.isInteger(value) && Number(value) >= 0)))
        : [],
    };
  },
  storage: async () => request<unknown[]>('/storage'),
  remoteTraining: {
    metrics: async (): Promise<RemoteServerMetrics> =>
      normalizeRemoteServerMetrics(await request<unknown>('/remote-training/metrics')),
  },
  datasets: {
    directories: (path?: string): Promise<DirectoryListing> =>
      request<DirectoryListing>(`/datasets/directories${path ? `?path=${encodeURIComponent(path)}` : ''}`),
    validate: async (path: string, datasetId: string, datasetMix?: string): Promise<DatasetValidationResult> => {
      const raw = await request<Record<string, unknown>>('/datasets/validate', {
        method: 'POST',
        body: body({ path, dataset_id: datasetId, dataset_mix: datasetMix }),
      });
      return {
        valid: Boolean(raw.valid),
        path: stringValue(raw.path),
        normalized_root: stringValue(raw.normalized_root),
        dataset_id: stringValue(raw.dataset_id),
        dataset_mix: stringValue(raw.dataset_mix) || undefined,
        format: stringValue(raw.format, 'unknown'),
        dataset_count: numberValue(raw.dataset_count),
        episode_count: numberValue(raw.episode_count),
        parquet_count: numberValue(raw.parquet_count),
        issues: normalizeIssues(raw.issues),
      };
    },
    inspect: async (path: string): Promise<DatasetInspectionResult> => {
      const raw = await request<Record<string, unknown>>('/datasets/inspect', {
        method: 'POST',
        body: body({ path }),
      });
      const support = stringValue(raw.builder_support);
      return {
        valid: Boolean(raw.valid),
        path: stringValue(raw.path),
        format: stringValue(raw.format, 'unknown'),
        format_family: stringValue(raw.format_family, 'unknown'),
        compatible_loaders: Array.isArray(raw.compatible_loaders) ? raw.compatible_loaders.map(String) : [],
        builder_support: ['direct', 'mixture_only', 'inventory_only'].includes(support)
          ? support as DatasetInspectionResult['builder_support']
          : 'unsupported',
        builder_ready: Boolean(raw.builder_ready),
        dataset_count: numberValue(raw.dataset_count, 1),
        episode_count: numberValue(raw.episode_count),
        step_count: numberValue(raw.step_count),
        parquet_count: numberValue(raw.parquet_count),
        size_bytes: numberValue(raw.size_bytes),
        issues: normalizeIssues(raw.issues),
      };
    },
    list: (): Promise<DatasetRegistration[]> => request<DatasetRegistration[]>('/datasets'),
    get: (id: string): Promise<DatasetRegistration> => request<DatasetRegistration>(`/datasets/${encodeURIComponent(id)}`),
    register: (payload: {
      name: string;
      description?: string;
      path: string;
      storage_mode: 'reference' | 'managed_copy';
      visibility: 'private' | 'shared';
      dataset_id?: string;
      dataset_mix?: string;
    }): Promise<DatasetRegistration> => request<DatasetRegistration>('/datasets', { method: 'POST', body: body(payload) }),
    revalidate: (id: string): Promise<DatasetRegistration> => request<DatasetRegistration>(`/datasets/${encodeURIComponent(id)}/validate`, { method: 'POST' }),
    preview: (id: string): Promise<DatasetPreview> => request<DatasetPreview>(`/datasets/${encodeURIComponent(id)}/preview`),
    stats: (id: string): Promise<UtilityRun> => request<UtilityRun>(`/datasets/${encodeURIComponent(id)}/stats`, { method: 'POST' }),
    remove: (id: string): Promise<{ ok: boolean; data_deleted: boolean; path: string }> => request(`/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    mixtures: {
      list: (): Promise<DatasetMixture[]> => request<DatasetMixture[]>('/datasets/mixtures'),
      create: (payload: {
        name: string;
        description?: string;
        visibility: 'private' | 'shared';
        members: Array<{ registration_id: string; pattern?: string; weight: number; robot_type?: string; trajectory_limit?: number }>;
        options?: Record<string, unknown>;
      }): Promise<DatasetMixture> => request<DatasetMixture>('/datasets/mixtures', { method: 'POST', body: body(payload) }),
      update: (id: string, payload: Partial<DatasetMixture>): Promise<DatasetMixture> => request<DatasetMixture>(`/datasets/mixtures/${encodeURIComponent(id)}`, { method: 'PATCH', body: body(payload) }),
      remove: (id: string): Promise<void> => request<void>(`/datasets/mixtures/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    },
  },
  resources: {
    list: (): Promise<{ items: ResourceRecord[]; hf_download_token: WandbSecretStatus }> => request('/resources'),
    install: (id: string, targetRoot?: string): Promise<UtilityRun> => request<UtilityRun>(`/resources/${encodeURIComponent(id)}/install`, { method: 'POST', body: body({ target_root: targetRoot || undefined }) }),
    registerPath: (id: string, path: string): Promise<ResourceRecord> => request<ResourceRecord>(`/resources/${encodeURIComponent(id)}/path`, { method: 'PUT', body: body({ path }) }),
    preprocess: (id: string, payload: { kind: string; inputs: Record<string, string>; gpu_count: number; gpu_ids?: number[] }): Promise<UtilityRun> => request<UtilityRun>(`/resources/${encodeURIComponent(id)}/preprocess`, { method: 'POST', body: body(payload) }),
  },
  utilities: {
    list: (): Promise<UtilityRun[]> => request<UtilityRun[]>('/utilities'),
    cancel: (id: string): Promise<UtilityRun> => request<UtilityRun>(`/utilities/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
    logUrl: (id: string) => `${API_BASE}/utilities/${encodeURIComponent(id)}/log`,
    outputUrl: (id: string) => `${API_BASE}/utilities/${encodeURIComponent(id)}/output`,
  },
  templates: {
    list: async (): Promise<Template[]> => recordArray(await request<unknown[]>('/templates')).map(normalizeTemplate),
    get: async (id: string): Promise<Template> => {
      const rows = recordArray(await request<unknown[]>('/templates')).map(normalizeTemplate);
      const found = rows.find((item) => item.id === id);
      if (!found) throw new ApiError('template_not_found', 404);
      return found;
    },
    create: async (payload: Partial<Template>, allowDuplicate = false): Promise<Template> => {
      const spec = payload.spec;
      if (!spec) throw new ApiError('template_spec_required', 422);
      const raw = await request<unknown>('/templates', {
        method: 'POST',
        body: body({
          name: payload.name,
          description: payload.description ?? '',
          visibility: payload.visibility === 'shared' ? 'shared' : 'private',
          spec: toBackendSpec(spec),
          ...(allowDuplicate ? { allow_duplicate: true } : {}),
        }),
      });
      return normalizeTemplate(raw);
    },
    update: async (id: string, payload: Partial<Template>, allowDuplicate = false): Promise<Template> => {
      const update: Record<string, unknown> = {};
      if (payload.name != null) update.name = payload.name;
      if (payload.description != null) update.description = payload.description;
      if (payload.visibility != null) update.visibility = payload.visibility === 'shared' ? 'shared' : 'private';
      if (payload.spec) update.spec = toBackendSpec(payload.spec);
      if (allowDuplicate) update.allow_duplicate = true;
      return normalizeTemplate(await request<unknown>(`/templates/${encodeURIComponent(id)}`, { method: 'PATCH', body: body(update) }));
    },
    remove: (id: string) => request<void>(`/templates/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  experiments: {
    list: async (): Promise<Experiment[]> => recordArray(await request<unknown[]>('/experiments')).map(normalizeExperiment),
    get: async (id: string): Promise<Experiment> => normalizeExperiment(await request<unknown>(`/experiments/${encodeURIComponent(id)}`)),
    resolve: async (spec: ExperimentSpec): Promise<ResolveResult> =>
      normalizeResolve(await request<unknown>('/experiments/resolve', { method: 'POST', body: body(experimentRequest(spec)) })),
    preflight: async (spec: ExperimentSpec): Promise<PreflightResult> => {
      const raw = await request<Record<string, unknown>>('/experiments/preflight', { method: 'POST', body: body(experimentRequest(spec)) });
      return { ok: Boolean(raw.can_submit), items: normalizeIssues(raw.issues) };
    },
    submit: async (spec: ExperimentSpec): Promise<{ experiment: Experiment; jobs: Job[] }> => {
      const raw = await request<Record<string, unknown>>('/experiments/submit', { method: 'POST', body: body(experimentRequest(spec)) });
      const id = stringValue(raw.experiment_id);
      return {
        experiment: { id, name: spec.name, status: 'queued' },
        jobs: recordArray(raw.jobs).map(normalizeJob),
      };
    },
  },
  jobs: {
    list: async (params?: URLSearchParams): Promise<Job[]> =>
      recordArray(await request<unknown[]>(`/jobs${params?.size ? `?${params.toString()}` : ''}`)).map(normalizeJob),
    get: async (id: string): Promise<Job> => normalizeJob(await request<unknown>(`/jobs/${encodeURIComponent(id)}`)),
    cancel: async (id: string): Promise<Job> => normalizeJob(await request<unknown>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' })),
    stop: async (id: string): Promise<Job> => normalizeJob(await request<unknown>(`/jobs/${encodeURIComponent(id)}/stop`, { method: 'POST' })),
    terminate: async (id: string): Promise<Job> => normalizeJob(await request<unknown>(`/jobs/${encodeURIComponent(id)}/terminate`, { method: 'POST' })),
    forceKill: async (id: string): Promise<Job> => normalizeJob(await request<unknown>(`/jobs/${encodeURIComponent(id)}/force-kill`, { method: 'POST' })),
    package: (id: string): Promise<UtilityRun> =>
      request<UtilityRun>(`/jobs/${encodeURIComponent(id)}/package`, { method: 'POST' }),
    remove: (id: string, confirmation: string): Promise<{ ok: boolean }> =>
      request(`/jobs/${encodeURIComponent(id)}`, { method: 'DELETE', body: body({ confirmation }) }),
    eventUrl: (id: string) => `${API_BASE}/jobs/${encodeURIComponent(id)}/events`,
  },
  workloads: {
    list: (params?: URLSearchParams): Promise<Workload[]> =>
      request<Workload[]>(`/workloads${params?.size ? `?${params.toString()}` : ''}`),
  },
  checkpoints: async (): Promise<Checkpoint[]> => recordArray(await request<unknown[]>('/checkpoints')).map(normalizeCheckpoint),
  checkpointDetail: async (id: string): Promise<CheckpointDetail> =>
    normalizeCheckpointDetail(await request<unknown>(`/checkpoints/${encodeURIComponent(id)}`)),
  mergeLoraCheckpoint: (id: string, payload: { model: string; output_name?: string; resources: { strategy: 'auto' | 'fixed'; gpu_count: number; gpu_ids?: number[] } }): Promise<UtilityRun> =>
    request<UtilityRun>(`/checkpoints/${encodeURIComponent(id)}/merge-lora`, { method: 'POST', body: body(payload) }),
  packageCheckpoint: (id: string): Promise<UtilityRun> =>
    request<UtilityRun>(`/checkpoints/${encodeURIComponent(id)}/package`, { method: 'POST' }),
  deleteCheckpoint: (id: string, confirmation: string) => request<void>(`/checkpoints/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    body: body({ confirmation }),
  }),
  deployments: {
    capabilities: async (): Promise<DeploymentCapabilities> =>
      normalizeDeploymentCapabilities(await request<unknown>('/deployment/capabilities')),
    preflight: async (payload: DeploymentCreateRequest): Promise<PreflightResult> => {
      const raw = await request<Record<string, unknown>>('/deployments/preflight', { method: 'POST', body: body(payload) });
      return {
        ok: Boolean(raw.ok ?? raw.can_submit),
        items: normalizeIssues(raw.items ?? raw.issues),
      };
    },
    create: async (payload: DeploymentCreateRequest): Promise<DeploymentMutationResult> =>
      normalizeDeploymentMutation(await request<unknown>('/deployments', { method: 'POST', body: body(payload) })),
    list: async (): Promise<ModelDeployment[]> => recordList(await request<unknown>('/deployments')).map(normalizeDeployment),
    get: async (id: string): Promise<ModelDeployment> =>
      normalizeDeployment(await request<unknown>(`/deployments/${encodeURIComponent(id)}`)),
    cancel: async (id: string): Promise<ModelDeployment> =>
      normalizeDeployment(await request<unknown>(`/deployments/${encodeURIComponent(id)}/cancel`, { method: 'POST' })),
    stop: async (id: string): Promise<ModelDeployment> =>
      normalizeDeployment(await request<unknown>(`/deployments/${encodeURIComponent(id)}/stop`, { method: 'POST' })),
    restart: async (id: string): Promise<DeploymentMutationResult> =>
      normalizeDeploymentMutation(await request<unknown>(`/deployments/${encodeURIComponent(id)}/restart`, { method: 'POST' })),
    rotateKey: async (id: string): Promise<DeploymentMutationResult> =>
      normalizeDeploymentMutation(await request<unknown>(`/deployments/${encodeURIComponent(id)}/rotate-key`, { method: 'POST' })),
    forceKill: async (id: string): Promise<ModelDeployment> =>
      normalizeDeployment(await request<unknown>(`/deployments/${encodeURIComponent(id)}/force-kill`, { method: 'POST' })),
    infer: async (id: string, input: PlaygroundInferenceInput): Promise<InferenceRun> => {
      const form = new FormData();
      form.set('instructions', JSON.stringify(input.instructions));
      if (input.states?.trim()) form.set('states', input.states.trim());
      form.set('save_inputs', String(input.save_inputs));
      input.images.forEach((image) => form.append('images', image, image.name));
      return normalizeInferenceRun(await request<unknown>(`/deployments/${encodeURIComponent(id)}/infer`, {
        method: 'POST',
        body: form,
      }));
    },
    eventUrl: (id: string) => `${API_BASE}/deployments/${encodeURIComponent(id)}/events`,
  },
  inferenceRuns: {
    list: async (deploymentId?: string): Promise<InferenceRun[]> => {
      const params = new URLSearchParams();
      if (deploymentId) params.set('deployment_id', deploymentId);
      const suffix = params.size ? `?${params.toString()}` : '';
      return recordList(await request<unknown>(`/inference-runs${suffix}`)).map(normalizeInferenceRun);
    },
    get: async (id: string): Promise<InferenceRun> =>
      normalizeInferenceRun(await request<unknown>(`/inference-runs/${encodeURIComponent(id)}`)),
  },
  publications: {
    list: async (checkpointId?: string): Promise<ModelPublication[]> => {
      const params = new URLSearchParams();
      if (checkpointId) params.set('checkpoint_id', checkpointId);
      const suffix = params.size ? `?${params.toString()}` : '';
      return recordList(await request<unknown>(`/model-publications${suffix}`)).map(normalizeModelPublication);
    },
    create: async (payload: { checkpoint_id: string; repo_id: string; revision: string; private: boolean }): Promise<ModelPublication> =>
      normalizeModelPublication(await request<unknown>('/model-publications', { method: 'POST', body: body(payload) })),
    get: async (id: string): Promise<ModelPublication> =>
      normalizeModelPublication(await request<unknown>(`/model-publications/${encodeURIComponent(id)}`)),
  },
  audit: {
    list: (limit = 200): Promise<AuditEvent[]> => request<AuditEvent[]>(`/audit?limit=${limit}`),
  },
  registry: {
    browse: (): Promise<RegistryBrowseResponse> => request<RegistryBrowseResponse>('/registry'),
    overlay: {
      get: (): Promise<RegistryOverlayState> => request<RegistryOverlayState>('/registry/overlay'),
      preview: (document: Record<string, unknown>): Promise<RegistryOverlayPreview> =>
        request<RegistryOverlayPreview>('/registry/overlay/preview', {
          method: 'POST',
          body: body(document),
        }),
      save: (document: Record<string, unknown>): Promise<RegistryOverlayState> =>
        request<RegistryOverlayState>('/registry/overlay', {
          method: 'PUT',
          body: body(document),
        }),
      remove: (): Promise<RegistryOverlayState> =>
        request<RegistryOverlayState>('/registry/overlay', { method: 'DELETE' }),
    },
  },
  referenceResults: {
    list: (): Promise<ReferenceResultsSnapshot> =>
      request<ReferenceResultsSnapshot>('/reference-results'),
    match: (
      benchmarkId: string,
      signature: Record<string, string | number | boolean>,
    ): Promise<ReferenceResultsMatch> =>
      request<ReferenceResultsMatch>('/reference-results/match', {
        method: 'POST',
        body: body({ benchmark_id: benchmarkId, signature }),
      }),
  },
  evaluations: {
    benchmarks: async (): Promise<EvaluationCapabilities> =>
      normalizeEvaluationCapabilities(await request<unknown>('/evaluation-benchmarks')),
    inspectCheckpoint: async (source: EvaluationCheckpointSource, combinationId?: string): Promise<EvaluationCheckpointInspection> =>
      normalizeEvaluationCheckpointInspection(await request<unknown>('/evaluations/inspect-checkpoint', {
        method: 'POST',
        body: body({ ...source, ...(combinationId ? { combination_id: combinationId } : {}) }),
      })),
    preflight: async (payload: EvaluationCreateRequest): Promise<PreflightResult> => {
      const raw = await request<Record<string, unknown>>('/evaluations/preflight', { method: 'POST', body: body(payload) });
      return {
        ok: Boolean(raw.ok ?? raw.can_submit),
        items: normalizeIssues(raw.items ?? raw.issues),
      };
    },
    create: async (payload: EvaluationCreateRequest): Promise<EvaluationCreateResult> => {
      const raw = await request<unknown>('/evaluations', { method: 'POST', body: body(payload) });
      const row = isRecord(raw) ? raw : {};
      const evaluation = normalizeEvaluation(row.evaluation ?? raw);
      return {
        evaluation,
        evaluations: recordArray(row.evaluations).map(normalizeEvaluation).length
          ? recordArray(row.evaluations).map(normalizeEvaluation)
          : [evaluation],
        group: isRecord(row.group) ? normalizeEvaluationGroup(row.group) : undefined,
        issues: normalizeIssues(row.issues),
      };
    },
    list: async (params?: URLSearchParams): Promise<EvaluationRun[]> =>
      recordList(await request<unknown>(`/evaluations${params?.size ? `?${params.toString()}` : ''}`)).map(normalizeEvaluation),
    get: async (id: string): Promise<EvaluationRun> =>
      normalizeEvaluation(await request<unknown>(`/evaluations/${encodeURIComponent(id)}`)),
    result: async (id: string): Promise<EvaluationResult> =>
      normalizeEvaluationResult(await request<unknown>(`/evaluations/${encodeURIComponent(id)}/result`), id),
    artifacts: async (id: string): Promise<EvaluationArtifact[]> => {
      const raw = await request<unknown>(`/evaluations/${encodeURIComponent(id)}/artifacts`);
      return recordList(raw).map((item) => normalizeEvaluationArtifact(item, id));
    },
    cancel: async (id: string): Promise<EvaluationRun> =>
      normalizeEvaluation(await request<unknown>(`/evaluations/${encodeURIComponent(id)}/cancel`, { method: 'POST' })),
    stop: async (id: string): Promise<EvaluationRun> =>
      normalizeEvaluation(await request<unknown>(`/evaluations/${encodeURIComponent(id)}/stop`, { method: 'POST' })),
    forceKill: async (id: string): Promise<EvaluationRun> =>
      normalizeEvaluation(await request<unknown>(`/evaluations/${encodeURIComponent(id)}/force-kill`, { method: 'POST' })),
    retryWandb: async (id: string): Promise<EvaluationRun> =>
      normalizeEvaluation(await request<unknown>(`/evaluations/${encodeURIComponent(id)}/wandb/retry`, { method: 'POST' })),
    compare: async (ids: string[]): Promise<EvaluationComparison> => {
      const raw = await request<unknown>('/evaluations/compare', { method: 'POST', body: body({ evaluation_ids: ids }) });
      const normalized = normalizeEvaluationComparison(raw);
      if (!normalized.items.length && Array.isArray(raw)) {
        return { compatible: true, warnings: [], items: raw.map((item) => ({ evaluation: normalizeEvaluation(item) })) };
      }
      return normalized;
    },
    eventUrl: (id: string) => `${API_BASE}/evaluations/${encodeURIComponent(id)}/events`,
    artifactUrl: (id: string, artifactId: string) =>
      `${API_BASE}/evaluations/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(artifactId)}`,
    resultDownloadUrl: (id: string) => `${API_BASE}/evaluations/${encodeURIComponent(id)}/result?download=1`,
    groups: {
      list: async (params?: URLSearchParams): Promise<EvaluationGroup[]> =>
        recordList(await request<unknown>(`/evaluation-groups${params?.size ? `?${params.toString()}` : ''}`)).map(normalizeEvaluationGroup),
      get: async (id: string): Promise<EvaluationGroup> =>
        normalizeEvaluationGroup(await request<unknown>(`/evaluation-groups/${encodeURIComponent(id)}`)),
      result: async (id: string): Promise<EvaluationResult> =>
        normalizeEvaluationResult(await request<unknown>(`/evaluation-groups/${encodeURIComponent(id)}/result`)),
      cancel: async (id: string): Promise<EvaluationGroup> =>
        normalizeEvaluationGroup(await request<unknown>(`/evaluation-groups/${encodeURIComponent(id)}/cancel`, { method: 'POST' })),
      resultDownloadUrl: (id: string) => `${API_BASE}/evaluation-groups/${encodeURIComponent(id)}/result?download=1`,
    },
  },
  settings: {
    get: async (): Promise<SystemSettings> => normalizeSettings(await request<unknown>('/settings')),
    update: async (payload: Partial<SystemSettings>): Promise<SystemSettings> =>
      normalizeSettings(await request<unknown>('/settings', { method: 'PATCH', body: body(settingsPayload(payload)) })),
    wandb: {
      status: async (): Promise<WandbSecretStatus> => {
        const row = await request<Record<string, unknown>>('/settings/wandb');
        return { configured: Boolean(row.configured) };
      },
      setApiKey: async (apiKey: string): Promise<WandbSecretStatus> => {
        const row = await request<Record<string, unknown>>('/settings/wandb/api-key', {
          method: 'PUT',
          body: body({ api_key: apiKey }),
        });
        return { configured: Boolean(row.configured) };
      },
      deleteApiKey: async (): Promise<WandbSecretStatus> => {
        const row = await request<Record<string, unknown>>('/settings/wandb/api-key', { method: 'DELETE' });
        return { configured: Boolean(row.configured) };
      },
    },
    huggingface: {
      downloadToken: {
        status: (): Promise<WandbSecretStatus> => request('/settings/huggingface/download-token'),
        set: (token: string): Promise<WandbSecretStatus> => request('/settings/huggingface/download-token', { method: 'PUT', body: body({ token }) }),
        remove: (): Promise<WandbSecretStatus> => request('/settings/huggingface/download-token', { method: 'DELETE' }),
      },
      publishToken: {
        status: (): Promise<WandbSecretStatus> => request('/users/me/huggingface/publish-token'),
        set: (token: string): Promise<WandbSecretStatus> => request('/users/me/huggingface/publish-token', { method: 'PUT', body: body({ token }) }),
        remove: (): Promise<WandbSecretStatus> => request('/users/me/huggingface/publish-token', { method: 'DELETE' }),
      },
    },
    updatePreferences: async (payload: Partial<User>): Promise<User> => normalizeUser(await request<unknown>('/users/me/preferences', {
      method: 'PATCH',
      body: body({
        language: payload.locale,
        theme: payload.theme,
        gpu_refresh_interval_seconds: payload.gpu_refresh_interval_seconds,
        experimental_enabled: payload.experimental_enabled,
      }),
    })),
  },
  users: {
    list: async (): Promise<User[]> => recordArray(await request<unknown[]>('/users')).map((item) => normalizeUser(item)),
    create: async (payload: Partial<User> & { password?: string }): Promise<User> => normalizeUser(await request<unknown>('/users', {
      method: 'POST',
      body: body({ username: payload.username, display_name: payload.display_name ?? '', password: payload.password, role: payload.role }),
    })),
    update: async (id: string, payload: Partial<User> & { password?: string }): Promise<User> => normalizeUser(await request<unknown>(`/users/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: body({ display_name: payload.display_name, password: payload.password || undefined, role: payload.role, is_active: payload.active }),
    })),
    remove: (id: string): Promise<void> => request<void>(`/users/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
};
