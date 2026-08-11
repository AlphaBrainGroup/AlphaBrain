export type Language = 'zh-CN' | 'en-US';
export type ThemeMode = 'light' | 'dark';
export type UserRole = 'administrator' | 'researcher';
export type JobStatus =
  | 'draft'
  | 'checking'
  | 'blocked'
  | 'queued'
  | 'starting'
  | 'running'
  | 'stopping'
  | 'completed'
  | 'stopped'
  | 'failed'
  | 'interrupted'
  | 'dependency_failed'
  | 'cancelled';

export interface SetupStatus {
  configured: boolean;
  mode?: 'personal' | 'laboratory';
}

export interface User {
  id: string;
  username: string;
  display_name?: string;
  role: UserRole;
  active?: boolean;
  locale?: Language;
  theme?: ThemeMode;
  gpu_refresh_interval_seconds?: number;
  experimental_enabled?: boolean;
  experimental_available?: boolean;
  deployment_mode?: 'personal' | 'laboratory';
  created_at?: string;
}

export interface GPU {
  id?: string;
  index: number;
  name: string;
  memory_used_mb: number;
  memory_total_mb: number;
  utilization_percent: number;
  temperature_c?: number;
  available?: boolean;
  external_processes?: number;
  job?: Pick<Job, 'id' | 'name' | 'owner_name' | 'status'>;
  error?: string;
}

export interface GPUInventory {
  available: boolean;
  error?: string;
  items: GPU[];
}

export interface TrainingTarget {
  mode: 'local' | 'remote';
  gpu_ids: number[];
}

export interface StorageSummary {
  path: string;
  mount_point?: string;
  used_bytes: number;
  total_bytes: number;
  free_bytes: number;
  warning?: boolean;
}

export interface SystemMetrics {
  available: boolean;
  cpu_percent?: number;
  load?: {
    one_minute: number;
    five_minutes: number;
    fifteen_minutes: number;
  };
  memory?: {
    total_bytes: number;
    used_bytes: number;
    available_bytes: number;
    percent: number;
  };
  error?: {
    code: string;
    message?: string;
  };
}

export interface RemoteTrainingStatus {
  enabled: boolean;
  target?: string;
}

export interface RemoteServerMetrics extends RemoteTrainingStatus {
  available: boolean;
  hostname?: string;
  collected_at?: string;
  stale?: boolean;
  gpus: GPU[];
  system_metrics?: SystemMetrics;
  storage?: StorageSummary;
  gpu_error?: string;
  error?: {
    code: string;
    message?: string;
  };
}

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  owner_id?: string;
  owner_name?: string;
  architecture?: string;
  method?: string;
  dataset?: string;
  status?: JobStatus;
  created_at?: string;
  updated_at?: string;
  job_id?: string;
  stages?: ExperimentStage[];
  config?: Record<string, unknown>;
}

export interface ExperimentStage {
  id: string;
  name: string;
  order: number;
  status: JobStatus;
  job_id?: string;
}

export interface Job {
  id: string;
  experiment_id?: string;
  name: string;
  owner_id?: string;
  owner_name?: string;
  status: JobStatus;
  queue_position?: number;
  gpu_ids?: number[];
  stage_name?: string;
  command?: string;
  pid?: number;
  progress?: number;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  error_summary?: string;
  latest_metrics?: Record<string, number>;
}

export interface Workload {
  id: string;
  kind: 'training' | 'deployment' | 'evaluation' | 'utility';
  subtype?: string;
  name: string;
  owner_name?: string;
  status: string;
  queue_scope: 'gpu_global' | 'cpu_utility';
  queue_position?: number;
  gpu_ids: number[];
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
  detail_url: string;
  can_delete?: boolean;
  can_package?: boolean;
  package_run?: UtilityRun;
}

export interface AuditEvent {
  id: string;
  actor_id?: string;
  action: string;
  target_type?: string;
  target_id?: string;
  detail: Record<string, unknown>;
  created_at: string;
}

export interface Checkpoint {
  id: string;
  experiment_id?: string;
  experiment_name?: string;
  owner_name?: string;
  path: string;
  step?: number;
  size_bytes?: number;
  kind?: string;
  created_at?: string;
  complete?: boolean;
  resumable?: boolean;
  important?: boolean;
  architecture_fingerprint?: string;
  best_score?: number;
  can_delete?: boolean;
  can_package?: boolean;
  deployable?: boolean;
  builtin?: boolean;
  name_i18n?: Partial<Record<Language, string>>;
  combination_id?: string;
  description?: string;
  description_i18n?: Partial<Record<Language, string>>;
  missing_requirements_i18n?: Partial<Record<Language, string[]>>;
  checkpoint_family?: string;
  checkpoint_format?: 'openpi' | 'lerobot' | 'alphabrain' | 'unknown' | string;
  checkpoint_format_label_i18n?: Partial<Record<Language, string>>;
  inspection_summary?: {
    format?: string;
    checkpoint_family?: string;
    checkpoint_format?: string;
    checkpoint_format_label?: Partial<Record<Language, string>>;
    framework?: string;
    combination_id?: string;
    issue_codes?: string[];
  };
}

export interface CheckpointDetail extends Checkpoint {
  name?: string;
  job_id?: string;
  metadata?: Record<string, unknown>;
  inspection: Record<string, unknown>;
  tools: {
    publish_huggingface: { available: boolean };
    merge_lora: {
      available: boolean;
      reason?: string;
      adapter_path?: string;
      action_model_path?: string;
      suggested_output_path?: string;
      output_exists?: boolean;
      models?: string[];
      experimental?: boolean;
    };
    add_qwen_special_tokens: { available: boolean; reason?: string };
  };
}

export interface Template {
  id: string;
  name: string;
  description?: string;
  visibility: 'personal' | 'shared';
  owner_id?: string;
  owner_name?: string;
  architecture?: string;
  method?: string;
  dataset?: string;
  version?: number;
  updated_at?: string;
  spec?: ExperimentSpec;
  builtin?: boolean;
  name_i18n?: Partial<Record<Language, string>>;
  description_i18n?: Partial<Record<Language, string>>;
  category?: string;
  tags?: string[];
  availability?: 'ready' | 'partial' | 'missing';
  requirements?: Array<{ id: string; path?: string; ready: boolean }>;
  recommended_gpu_count?: number;
}

export interface TemplateDuplicateMatch {
  id: string;
  name: string;
  owner_id?: string;
  owner_name?: string;
  visibility: 'personal' | 'shared';
  builtin: boolean;
  same_name: boolean;
  same_spec: boolean;
}

export interface TemplateDuplicateDetail {
  code: 'duplicate_template';
  matches: TemplateDuplicateMatch[];
}

export type CapabilityKind = 'backbone' | 'action_head' | 'method' | 'dataset';
export type CapabilityStatus = 'verified' | 'experimental' | 'unsupported';

export type WandbUploadCategory = 'metrics' | 'config' | 'system' | 'checkpoints' | 'videos';

export interface WandbCategoryCapability {
  id: WandbUploadCategory;
  supported: boolean;
  title: Partial<Record<Language, string>>;
  description: Partial<Record<Language, string>>;
  reason?: Partial<Record<Language, string>>;
}

/** Per-experiment W&B preferences. Credentials are deliberately not part of this type. */
export interface WandbRunConfig {
  enabled: boolean;
  mode: 'online' | 'offline' | 'disabled';
  project: string;
  entity: string;
  run_name: string;
  group: string;
  job_type: string;
  tags: string[];
  notes: string;
  categories: WandbUploadCategory[];
}

export interface WandbSecretStatus {
  configured: boolean;
}

export interface PretrainedDirectoryStatus {
  required: boolean;
  configured: boolean;
  path?: string;
  exists: boolean;
  issue?: {
    code: 'missing_environment_variable' | 'pretrained_directory_not_found' | 'pretrained_path_not_directory' | string;
    variables?: string[];
    message_i18n?: Partial<Record<Language, string>>;
  };
}

export interface Capability {
  id: string;
  kind: CapabilityKind;
  name: string;
  name_zh?: string;
  description?: string;
  description_zh?: string;
  status: CapabilityStatus;
  category?: 'vlm' | 'world_model' | string;
  compatible_with?: string[];
  recommended_gpu_count?: number;
  estimated_memory_gb?: number;
  requires_checkpoint?: boolean;
  pretrained_directory?: PretrainedDirectoryStatus;
  parameters?: ParameterSchema[];
}

export interface ParameterSchema {
  key: string;
  label?: string;
  label_zh?: string;
  description?: string;
  description_zh?: string;
  type: 'string' | 'number' | 'integer' | 'boolean' | 'select' | 'path' | 'password' | 'textarea';
  default?: unknown;
  required?: boolean;
  min?: number;
  max?: number;
  options?: Array<{ label: string; label_zh?: string; value: string | number }>;
}

export interface WorkflowField {
  key: string;
  type: ParameterSchema['type'];
  label?: Partial<Record<Language, string>>;
  description?: Partial<Record<Language, string>>;
  default?: unknown;
  required?: boolean;
  min?: number;
  max?: number;
  options?: Array<{ value: string | number; label?: Partial<Record<Language, string>> }>;
  visible_when?: { key: string; equals?: unknown; in?: unknown[] };
  methods?: string[];
  algorithms?: string[];
  config_path?: string;
  stage?: string;
}

export interface WorkflowSchema {
  id: string;
  schema_version: number;
  title?: Partial<Record<Language, string>>;
  description?: Partial<Record<Language, string>>;
  fields: WorkflowField[];
}

export interface WorkflowSpec {
  id: string;
  schema_version: number;
  config: Record<string, unknown>;
}

export interface ResumeSpec {
  mode: 'none' | 'weights_only' | 'full_state';
  checkpoint?: string;
}

export type DeploymentStatus = 'queued' | 'starting' | 'running' | 'stopping' | 'stopped' | 'failed' | 'cancelled';

export interface DeploymentAdapter {
  id: string;
  name: string;
  name_zh?: string;
  description?: string;
  description_zh?: string;
  protocol?: string;
  parameters: ParameterSchema[];
}

export interface DeploymentCombination {
  id: string;
  name: string;
  name_zh?: string;
  description?: string;
  description_zh?: string;
  status: CapabilityStatus;
  adapter_id: string;
  backbone_id: string;
  action_head_id: string;
  recommended_gpu_count?: number;
  parameters: ParameterSchema[];
}

export interface DeploymentCapabilities {
  adapters: DeploymentAdapter[];
  combinations: DeploymentCombination[];
  experimental_allowed?: boolean;
}

export type EvaluationStatus = 'queued' | 'starting' | 'running' | 'stopping' | 'completed' | 'failed' | 'stopped' | 'cancelled' | 'interrupted';
export type EvaluationPreset = 'quick' | 'standard' | 'full' | 'custom';
export type EvaluationKind = 'standard' | 'batch' | 'cl_matrix' | 'rl_iterations' | 'online_stdp' | 'world_model_video';
export type EvaluationSourceKind = 'temporary_checkpoint' | 'managed_deployment';

export interface EvaluationChoice {
  value: string;
  label: string;
  label_zh?: string;
  description?: string;
  description_zh?: string;
}

export interface EvaluationPresetDefinition {
  id: EvaluationPreset;
  name: string;
  name_zh?: string;
  description?: string;
  description_zh?: string;
  parameters: Record<string, unknown>;
}

export interface EvaluationCompatibility {
  combination_ids: string[];
  backbone_ids?: string[];
  action_head_ids?: string[];
}

export interface EvaluationEnvironmentCheck {
  id: string;
  configured: boolean;
  required?: boolean;
  title?: string;
  title_zh?: string;
  message?: string;
  message_zh?: string;
}

export interface EvaluationBenchmark {
  id: string;
  name: string;
  name_zh?: string;
  description?: string;
  description_zh?: string;
  status: CapabilityStatus;
  available: boolean;
  suites: EvaluationChoice[];
  task_sets: EvaluationChoice[];
  splits: EvaluationChoice[];
  presets: EvaluationPresetDefinition[];
  defaults: Record<string, unknown>;
  parameters: ParameterSchema[];
  compatibility: EvaluationCompatibility;
  environment: EvaluationEnvironmentCheck[];
}

export interface EvaluationCapabilities {
  benchmarks: EvaluationBenchmark[];
  experimental_allowed?: boolean;
}

export type EvaluationCheckpointSource =
  | { kind: 'indexed'; checkpoint_id: string }
  | { kind: 'local'; path: string };

export interface EvaluationCheckpointDetection {
  framework?: string;
  backbone?: string;
  action_head?: string;
  adapter_id?: string;
  combination_id?: string;
  candidate_combination_ids: string[];
}

export interface EvaluationCheckpointCandidate {
  combination_id: string;
  benchmark_ids: string[];
}

export interface EvaluationCheckpointInspection {
  valid?: boolean;
  detected: EvaluationCheckpointDetection;
  evaluation_candidates: EvaluationCheckpointCandidate[];
  compatible_benchmark_ids: string[];
  issues: PreflightItem[];
}

export interface EvaluationWandbConfig {
  enabled: boolean;
  mode: 'online' | 'offline';
  project: string;
  entity: string;
  run_name: string;
  tags: string[];
  categories: Array<'summary' | 'tasks' | 'config' | 'videos'>;
}

export interface EvaluationCreateRequest {
  name: string;
  kind?: EvaluationKind;
  source_kind?: EvaluationSourceKind;
  checkpoint_source?: EvaluationCheckpointSource;
  checkpoint_sources?: EvaluationCheckpointSource[];
  suites?: string[];
  deployment_id?: string;
  combination_id?: string;
  benchmark_id: string;
  preset: EvaluationPreset;
  suite?: string;
  task_set?: string;
  split?: string;
  parameters: Record<string, unknown>;
  model_parameters?: Record<string, unknown>;
  resources: {
    strategy: 'auto' | 'fixed';
    gpu_count: number;
    gpu_ids?: number[];
  };
  wandb: EvaluationWandbConfig;
  acknowledge_experimental?: boolean;
}

export interface EvaluationTaskMetric {
  id: string;
  name?: string;
  suite?: string;
  num_successes: number;
  num_episodes: number;
  success_rate: number;
  duration_seconds?: number;
  metadata?: Record<string, unknown>;
}

export interface EvaluationEpisodeMetric {
  id: string;
  task_id?: string;
  task_name?: string;
  episode_index?: number;
  seed?: number;
  success: boolean;
  duration_seconds?: number;
  steps?: number;
  video_id?: string;
}

export interface EvaluationVideo {
  id: string;
  name: string;
  task_id?: string;
  task_name?: string;
  episode_index?: number;
  success?: boolean;
  path?: string;
  url?: string;
  size_bytes?: number;
}

export interface EvaluationArtifact {
  id?: string;
  kind: string;
  path: string;
  name: string;
  media_type?: string;
  url?: string;
  size_bytes?: number;
  metadata: Record<string, unknown>;
}

export interface EvaluationMatrix {
  id: string;
  name: string;
  row_labels: string[];
  column_labels: string[];
  values: Array<Array<number | null>>;
  value_format: string;
  metadata: Record<string, unknown>;
}

export interface EvaluationSeriesPoint {
  x: string | number;
  y: number | null;
  metadata: Record<string, unknown>;
}

export interface EvaluationSeries {
  id: string;
  name: string;
  x_label: string;
  y_label: string;
  points: EvaluationSeriesPoint[];
  metadata: Record<string, unknown>;
}

export interface EvaluationChildResult {
  evaluation_id: string;
  name: string;
  position: number;
  kind: string;
  status: EvaluationStatus | string;
  checkpoint?: string;
  suite?: string;
  result_path?: string;
  summary: Record<string, unknown>;
  error?: string;
}

export interface EvaluationResult {
  schema_version: string;
  kind?: EvaluationKind | string;
  status?: string;
  evaluation_id?: string;
  benchmark_id?: string;
  summary: {
    num_tasks?: number;
    success_rate: number;
    num_successes: number;
    num_episodes: number;
  } & Record<string, unknown>;
  duration_seconds?: number;
  tasks: EvaluationTaskMetric[];
  episodes: EvaluationEpisodeMetric[];
  videos: EvaluationVideo[];
  artifacts: EvaluationArtifact[];
  matrices: EvaluationMatrix[];
  series: EvaluationSeries[];
  comparisons: Array<Record<string, unknown>>;
  children: EvaluationChildResult[];
  parameters: Record<string, unknown>;
  metadata: Record<string, unknown>;
  error?: string;
}

export interface EvaluationRun {
  id: string;
  group_id?: string;
  position?: number;
  evaluation_kind?: string;
  source_kind?: EvaluationSourceKind | string;
  deployment_id?: string;
  result_schema_version?: string;
  name: string;
  owner_id?: string;
  owner_name?: string;
  status: EvaluationStatus;
  queue_position?: number;
  checkpoint_id?: string;
  checkpoint_path?: string;
  combination_id: string;
  adapter_id?: string;
  backbone_id?: string;
  action_head_id?: string;
  benchmark_id: string;
  preset: EvaluationPreset;
  suite?: string;
  task_set?: string;
  split?: string;
  parameters: Record<string, unknown>;
  wandb: EvaluationWandbConfig;
  wandb_status?: string;
  wandb_error?: string;
  wandb_run_url?: string;
  requested_gpu_ids: number[];
  assigned_gpu_ids: number[];
  requested_gpu_count?: number;
  progress?: number;
  phase?: string;
  output_path?: string;
  log_path?: string;
  result_path?: string;
  pid?: number;
  exit_code?: number;
  error_summary?: string;
  result?: EvaluationResult;
  videos: EvaluationVideo[];
  queued_at?: string;
  started_at?: string;
  finished_at?: string;
  created_at?: string;
}

export interface EvaluationGroup {
  id: string;
  owner_id?: string;
  owner_name?: string;
  name: string;
  kind: EvaluationKind | string;
  status: EvaluationStatus | 'partial' | string;
  spec: Record<string, unknown>;
  result_summary: Record<string, unknown>;
  result_path?: string;
  error?: string;
  runs: EvaluationRun[];
  result?: EvaluationResult;
  created_at?: string;
  updated_at?: string;
  started_at?: string;
  finished_at?: string;
}

export interface EvaluationCreateResult {
  evaluation: EvaluationRun;
  evaluations: EvaluationRun[];
  group?: EvaluationGroup;
  issues: PreflightItem[];
}

export interface EvaluationStreamEvent {
  id?: string;
  type: 'status' | 'log' | 'progress' | 'result' | 'heartbeat' | 'end' | string;
  timestamp?: string;
  data?: unknown;
  status?: EvaluationStatus;
  line?: string;
  progress?: number;
  phase?: string;
}

export interface EvaluationComparisonItem {
  evaluation: EvaluationRun;
  result?: EvaluationResult;
}

export interface EvaluationComparison {
  benchmark_id?: string;
  signature?: string | { suite?: string; task_set?: string; split?: string };
  compatible: boolean;
  warnings: string[];
  items: EvaluationComparisonItem[];
}

export type DeploymentCheckpointSource =
  | { kind: 'indexed'; checkpoint_id: string }
  | { kind: 'local'; path: string };

export interface DeploymentCreateRequest {
  name: string;
  checkpoint_source: DeploymentCheckpointSource;
  combination_id: string;
  resources: {
    strategy: 'auto' | 'fixed';
    gpu_count: number;
    gpu_ids?: number[];
  };
  endpoint: {
    scope: 'local' | 'lan';
    advertised_host?: string;
    port?: number;
    idle_timeout_seconds: number;
  };
  parameters: Record<string, unknown>;
  acknowledge_experimental?: boolean;
}

export interface DeploymentEndpoint {
  scope: 'local' | 'lan';
  bind_host?: string;
  advertised_host?: string;
  port?: number;
  url?: string;
}

export interface ModelDeployment {
  id: string;
  name: string;
  owner_id?: string;
  owner_name?: string;
  status: DeploymentStatus;
  queue_position?: number;
  checkpoint_id?: string;
  checkpoint_path?: string;
  combination_id: string;
  adapter_id?: string;
  backbone_id?: string;
  action_head_id?: string;
  requested_gpu_count: number;
  requested_gpu_ids: number[];
  assigned_gpu_ids: number[];
  endpoint: DeploymentEndpoint;
  idle_timeout_seconds: number;
  parameters: Record<string, unknown>;
  log_path?: string;
  pid?: number;
  error_summary?: string;
  queued_at?: string;
  started_at?: string;
  ready_at?: string;
  finished_at?: string;
  created_at?: string;
  managed_inference_available?: boolean;
}

export interface DeploymentMutationResult {
  deployment: ModelDeployment;
  api_key?: string;
}

export interface DeploymentStreamEvent {
  id?: string;
  type: 'status' | 'log' | 'heartbeat' | 'end' | string;
  timestamp?: string;
  data?: unknown;
  status?: DeploymentStatus;
  line?: string;
}

export type InferenceRunStatus = 'running' | 'completed' | 'failed' | 'timed_out';

export interface InferenceRun {
  id: string;
  request_id: string;
  deployment_id?: string;
  owner_id?: string;
  status: InferenceRunStatus;
  batch_size: number;
  image_count: number;
  save_inputs: boolean;
  inputs_saved: boolean;
  request_summary: Record<string, unknown>;
  deployment_metadata: Record<string, unknown>;
  output?: Record<string, unknown>;
  latency_ms?: number;
  error?: string;
  created_at?: string;
  finished_at?: string;
}

export interface PlaygroundInferenceInput {
  instructions: string[];
  states?: string;
  images: File[];
  save_inputs: boolean;
}

export interface ModelPublication {
  id: string;
  owner_id: string;
  checkpoint_id?: string;
  utility_run_id?: string;
  provider: 'huggingface';
  repo_id: string;
  revision: string;
  private: boolean;
  status: string;
  source_path: string;
  result_url?: string;
  metadata: Record<string, unknown>;
  error?: string;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  updated_at?: string;
}

export interface CapabilityResponse {
  capabilities?: Capability[];
  backbones?: Capability[];
  action_heads?: Capability[];
  methods?: Capability[];
  datasets?: Capability[];
  experimental_allowed?: boolean;
  user_experimental_enabled?: boolean;
  combinations?: Array<{
    id: string;
    status: CapabilityStatus;
    backbone: string;
    action_head: string;
    method: string;
    datasets: string[];
    min_gpus?: number;
    launcher?: string;
    rl_algorithm?: string;
    workflow: WorkflowSchema;
    wandb_categories: WandbCategoryCapability[];
  }>;
}

export interface ExperimentSpec {
  spec_version?: 2;
  name: string;
  description?: string;
  template_id?: string;
  backbone_id?: string;
  action_head_id?: string;
  method_id?: string;
  dataset_id?: string;
  dataset_path?: string;
  dataset_registration_id?: string;
  dataset_mixture_id?: string;
  checkpoint_path?: string;
  workflow?: WorkflowSpec;
  resume?: ResumeSpec;
  parameters: Record<string, unknown>;
  resources: {
    strategy: 'auto' | 'fixed';
    gpu_count: number;
    gpu_ids?: number[];
  };
  wandb?: WandbRunConfig;
  expert_overrides?: Record<string, unknown>;
  experimental_confirmed?: boolean;
}

export interface DirectoryListing {
  path: string;
  parent?: string;
  directories: Array<{ name: string; path: string }>;
}

export interface DatasetValidationResult {
  valid: boolean;
  path: string;
  normalized_root: string;
  dataset_id: string;
  dataset_mix?: string;
  format: string;
  dataset_count: number;
  episode_count: number;
  parquet_count: number;
  issues: PreflightItem[];
}

export type UtilityStatus = 'queued' | 'starting' | 'running' | 'stopping' | 'completed' | 'failed' | 'cancelled' | 'stopped' | 'interrupted';

export interface UtilityRun {
  id: string;
  owner_id?: string;
  kind: string;
  resource_id?: string;
  status: UtilityStatus;
  queue_class: 'cpu' | 'gpu';
  requested_gpu_count: number;
  requested_gpu_ids: number[];
  assigned_gpu_ids: number[];
  parameters?: Record<string, unknown>;
  output_path?: string;
  log_path?: string;
  error?: string;
  exit_code?: number;
  queue_position?: number;
  progress?: {
    percent: number;
    phase?: string;
    current?: number;
    total?: number;
    unit?: 'bytes' | 'files' | string;
    message?: string;
  };
  queued_at?: string;
  started_at?: string;
  finished_at?: string;
}

export interface ResourceRecord {
  id: string;
  kind: 'pretrained_model' | 'dataset' | 'world_model' | string;
  name: string;
  source?: string | string[];
  installable: boolean;
  registerable?: boolean;
  requires_admin?: boolean;
  requires_hf_token?: boolean;
  target_path?: string;
  install_root?: string;
  status: 'installed' | 'missing' | 'unconfigured' | string;
  preprocess?: Array<'t5' | 'reason1' | 'umt5' | 'reason1_projection'>;
  dependencies?: string[];
  active_run?: UtilityRun;
}

export interface DatasetRegistration {
  id: string;
  owner_id: string;
  owner_name?: string;
  name: string;
  description?: string;
  path: string;
  source_path: string;
  storage_mode: 'reference' | 'managed_copy';
  visibility: 'private' | 'shared';
  format: string;
  status: 'ready' | 'copying' | 'invalid' | string;
  dataset_id: string;
  dataset_mix?: string;
  fingerprint?: string;
  size_bytes: number;
  episode_count: number;
  step_count: number;
  validation?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  copy_run_id?: string;
  stats_run_id?: string;
  stats_status?: string;
  created_at?: string;
  updated_at?: string;
}

export type DatasetBuilderSupport = 'direct' | 'mixture_only' | 'inventory_only' | 'unsupported';

export interface DatasetInspectionResult {
  valid: boolean;
  path: string;
  format: string;
  format_family: string;
  compatible_loaders: string[];
  builder_support: DatasetBuilderSupport;
  builder_ready: boolean;
  dataset_count: number;
  episode_count: number;
  step_count: number;
  parquet_count: number;
  size_bytes: number;
  issues: PreflightItem[];
}

export interface DatasetMixtureMember {
  registration_id: string;
  pattern?: string;
  weight: number;
  robot_type?: string;
  trajectory_limit?: number;
  dataset_path?: string;
  dataset_name?: string;
  dataset_status?: string;
  dataset_format?: string;
  format_family?: string;
  builder_support?: DatasetBuilderSupport;
}

export interface DatasetMixture {
  id: string;
  owner_id: string;
  owner_name?: string;
  name: string;
  description?: string;
  visibility: 'private' | 'shared';
  members: DatasetMixtureMember[];
  resolved_members: DatasetMixtureMember[];
  mixture_spec: Array<Record<string, unknown>>;
  options: Record<string, unknown>;
  version: number;
  created_at?: string;
  updated_at?: string;
}

export interface DatasetPreview {
  summary: Record<string, unknown>;
  episodes: Array<Record<string, unknown>>;
  tasks: Array<Record<string, unknown>>;
}

export interface ResolveResult {
  resolved_config: Record<string, unknown>;
  command?: string;
  diff?: Array<{ path: string; before?: unknown; after?: unknown; source?: string }>;
  warnings?: string[];
}

export interface PreflightItem {
  id?: string;
  level: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message?: string;
  message_i18n?: Partial<Record<Language, string>>;
  path?: string;
  detail?: Record<string, unknown>;
}

export interface PreflightResult {
  ok: boolean;
  items: PreflightItem[];
}

export interface DashboardData {
  running_jobs?: number;
  queued_jobs?: number;
  failed_jobs?: number;
  completed_jobs?: number;
  recent_experiments?: Experiment[];
  alerts?: Array<{ id?: string; level: 'warning' | 'error' | 'info'; title: string; message?: string }>;
  storage?: StorageSummary;
  system_metrics?: SystemMetrics;
  /** Compatibility alias for servers that expose the snapshot as `system`. */
  system?: SystemMetrics;
  remote_training?: RemoteTrainingStatus;
}

export interface MetricPoint {
  timestamp?: number | string;
  step?: number;
  iteration?: number;
  phase?: string;
  metrics: Record<string, number>;
}

export interface JobStreamEvent {
  id?: string;
  type: 'status' | 'log' | 'metric' | 'heartbeat' | string;
  timestamp?: string;
  data?: unknown;
  status?: JobStatus;
  line?: string;
  metric?: MetricPoint;
}

export interface RegistryOverlayStatus {
  configured: boolean;
  sha256?: string | null;
  overlay_version?: string | null;
  component_additions: number;
  combination_additions: number;
  deployment_combination_additions: number;
  disabled_entries: number;
}

export interface RegistryComponentView {
  id: string;
  category: 'backbones' | 'action_heads' | 'training_methods' | 'datasets';
  kind?: string | null;
  status: CapabilityStatus;
  label: Partial<Record<Language, string>>;
  description: Partial<Record<Language, string>>;
  origin: 'built_in' | 'overlay';
  config_reference?: string | null;
}

export interface RegistryCombinationView {
  id: string;
  status: CapabilityStatus;
  origin: 'built_in' | 'overlay';
  backbone: string;
  action_head: string;
  method: string;
  datasets: string[];
  launcher?: string;
  workflow?: string;
  configuration_references?: string[];
  min_gpus: number;
  valid_references: boolean;
}

export interface RegistryDeploymentAdapterView {
  id: string;
  status: CapabilityStatus;
  label: Partial<Record<Language, string>>;
  entrypoint?: string;
  transport?: string;
  request_format?: string;
}

export interface RegistryDeploymentCombinationView {
  id: string;
  status: CapabilityStatus;
  origin: 'built_in' | 'overlay';
  adapter: string;
  backbone: string;
  action_head: string;
  source_combinations: string[];
  benchmarks: string[];
  recommended_gpu_count: number;
  valid_references: boolean;
}

export interface RegistryView {
  schema: 'alphabrain.registry-view';
  schema_version: 1;
  registry_version: string;
  components: RegistryComponentView[];
  combinations: RegistryCombinationView[];
  deployment_adapters: RegistryDeploymentAdapterView[];
  deployment_combinations: RegistryDeploymentCombinationView[];
  workflow_ids: string[];
  warnings: Array<{ code: string; id?: string; references?: string[] }>;
  summary: {
    component_count: number;
    combination_count: number;
    overlay_component_count: number;
    overlay_combination_count: number;
    deployment_adapter_count: number;
    deployment_combination_count: number;
    overlay_deployment_combination_count: number;
  };
}

export interface RegistryBrowseResponse {
  view: RegistryView;
  overlay_status: RegistryOverlayStatus;
}

export interface RegistryOverlayState {
  status: RegistryOverlayStatus;
  overlay: Record<string, unknown> | null;
}

export interface RegistryOverlayPreview {
  overlay: Record<string, unknown>;
  view: RegistryView;
}

export interface ReferenceMetricDefinition {
  id: string;
  label: string;
  unit: 'percent' | 'count' | 'steps' | 'score' | 'seconds';
  direction: 'higher' | 'lower' | 'neutral';
}

export interface ReferenceResultRow {
  id: string;
  label: string;
  origin: string;
  values: Record<string, number | null>;
  context: Record<string, string | number | boolean>;
}

export interface ReferenceResultSeries {
  id: string;
  label: string;
  origin: string;
  points: Array<{ step: number; values: Record<string, number | null> }>;
  context: Record<string, string | number | boolean>;
}

export interface ReferenceResultSet {
  id: string;
  group: string;
  category: string;
  benchmark_id: string;
  signature: Record<string, string | number | boolean>;
  title: string;
  kind: 'table' | 'series';
  metrics: ReferenceMetricDefinition[];
  rows: ReferenceResultRow[];
  series: ReferenceResultSeries[];
  notes: string;
}

export interface ReferenceResultsSnapshot {
  schema: 'alphabrain.reference-results';
  schema_version: 1;
  version: string;
  source_url: string;
  source_asset: string;
  captured_at: string;
  title: string;
  disclaimer: string;
  result_sets: ReferenceResultSet[];
}

export interface ReferenceResultsMatch {
  schema: 'alphabrain.reference-results-match';
  schema_version: 1;
  snapshot: Pick<ReferenceResultsSnapshot, 'version' | 'source_url' | 'source_asset' | 'captured_at' | 'title' | 'disclaimer'>;
  benchmark_id: string;
  signature: Record<string, string | number | boolean>;
  items: ReferenceResultSet[];
}

export interface SystemSettings {
  mode?: 'personal' | 'laboratory';
  experimental_allowed?: boolean;
  results_roots?: string[];
  /** Compatibility alias for older forms; new UI uses `results_roots`. */
  results_root?: string;
  storage_monitor_path?: string;
  dataset_roots?: string[];
  managed_dataset_root?: string;
  pretrained_root?: string;
  cpu_utility_concurrency?: number;
  python_path?: string;
  model_server_python?: string;
  remote_training_enabled?: boolean;
  remote_training_host?: string;
  remote_training_user?: string;
  remote_training_port?: number;
  remote_training_repo_root?: string;
  remote_training_identity_file?: string;
  remote_training_gpu_ids?: number[];
  remote_training_setup_command?: string;
  secure_cookies?: boolean;
  secure_cookies_locked?: boolean;
  low_disk_percent?: number;
  low_disk_bytes?: number;
  low_disk_gib?: number;
  stop_grace_seconds?: number;
  admin_password?: string;
  [key: string]: unknown;
}

export interface Paged<T> {
  items: T[];
  total?: number;
  page?: number;
  page_size?: number;
}
