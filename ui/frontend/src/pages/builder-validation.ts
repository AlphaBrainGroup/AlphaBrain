import type { TrainingTarget, WandbRunConfig, WorkflowField } from '../api/types';

interface WizardValues {
  name?: string;
  backbone_id?: string;
  action_head_id?: string;
  method_id?: string;
  dataset_id?: string;
  dataset_source?: 'builtin' | 'registered' | 'mixture' | 'local';
  dataset_registration_id?: string;
  dataset_mixture_id?: string;
  use_local_dataset?: boolean;
  dataset_path?: string;
  batch_size?: number;
  learning_rate?: number;
  max_steps?: number;
  save_interval?: number;
  resource_strategy?: 'auto' | 'fixed';
  gpu_count?: number;
  gpu_ids?: number[];
  checkpoint_path?: string;
  resume_mode?: 'none' | 'weights_only' | 'full_state';
  resume_checkpoint?: string;
  module_parameters?: Record<string, unknown>;
  workflow_config?: Record<string, unknown>;
  wandb?: {
    enabled?: boolean;
    mode?: 'online' | 'offline' | 'disabled';
    project?: string;
    categories?: string[];
  };
}

export function defaultWandbRunConfig(): WandbRunConfig {
  return {
    enabled: false,
    mode: 'online',
    project: 'AlphaBrain',
    entity: '',
    run_name: '',
    group: '',
    job_type: '',
    tags: [],
    notes: '',
    categories: ['metrics', 'config', 'system'],
  };
}

export interface BuilderResources {
  strategy: 'auto' | 'fixed';
  gpu_count: number;
  gpu_ids?: number[];
}

export function builderGpuIds(
  localGpuIds: number[],
  trainingTarget: TrainingTarget | undefined,
): number[] {
  const source = trainingTarget?.mode === 'remote'
    ? trainingTarget.gpu_ids
    : localGpuIds;
  return Array.from(new Set(source.filter((value) => Number.isInteger(value) && value >= 0)));
}

function present(value: unknown): boolean {
  return value !== undefined && value !== null && (typeof value !== 'string' || Boolean(value.trim()));
}

function positive(value: unknown): boolean {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

export function isBuilderStepComplete(
  step: number,
  values: WizardValues,
  options: {
    localDatasetValid: boolean;
    datasetReferenceReady?: boolean;
    requiresCheckpoint: boolean;
    requiredParameterKeys: string[];
    requiredWorkflowKeys?: string[];
    expertYamlValid: boolean;
    singleGpu?: boolean;
    supportedWandbCategoryIds?: string[];
  },
): boolean {
  if (step === 0) {
    return Boolean(values.backbone_id && values.action_head_id && values.method_id && values.dataset_id)
      && (options.datasetReferenceReady ?? (!values.use_local_dataset || (present(values.dataset_path) && options.localDatasetValid)));
  }
  if (step === 1) {
    const fixedGpuReady = options.singleGpu
      || values.resource_strategy !== 'fixed'
      || Boolean(values.gpu_ids?.length && values.gpu_ids.length === values.gpu_count);
    const resourceReady = options.singleGpu
      || (Boolean(values.resource_strategy) && positive(values.gpu_count));
    const checkpointReady = !options.requiresCheckpoint || present(values.checkpoint_path);
    const resumeReady = !values.resume_mode || values.resume_mode === 'none' || present(values.resume_checkpoint);
    const dynamicReady = options.requiredParameterKeys.every((key) => present(values.module_parameters?.[key]));
    const workflowReady = (options.requiredWorkflowKeys ?? []).every((key) => present(values.workflow_config?.[key]));
    const wandbReady = !values.wandb?.enabled || (
      ['online', 'offline'].includes(values.wandb.mode ?? '')
      && present(values.wandb.project)
      && (values.wandb.categories ?? []).every((id) => options.supportedWandbCategoryIds?.includes(id))
    );
    return Boolean(values.name?.trim() && values.name.trim().length >= 2)
      && positive(values.batch_size)
      && positive(values.learning_rate)
      && positive(values.max_steps)
      && positive(values.save_interval)
      && resourceReady
      && fixedGpuReady
      && checkpointReady
      && resumeReady
      && dynamicReady
      && workflowReady
      && wandbReady;
  }
  if (step === 2) return options.expertYamlValid;
  return true;
}

export function workflowFieldActive(
  field: WorkflowField,
  values: Record<string, unknown> | undefined,
  context: { method?: string; algorithm?: string } = {},
): boolean {
  if (field.methods?.length && !field.methods.includes(context.method ?? '')) return false;
  if (field.algorithms?.length && !field.algorithms.includes(context.algorithm ?? '')) return false;
  if (!field.visible_when) return true;
  const current = values?.[field.visible_when.key];
  if ('equals' in field.visible_when) return current === field.visible_when.equals;
  if (field.visible_when.in) return field.visible_when.in.includes(current);
  return true;
}

export function normalizeBuilderResources(
  values: Pick<WizardValues, 'resource_strategy' | 'gpu_count' | 'gpu_ids'>,
  detectedGpuCount?: number,
): BuilderResources {
  if (detectedGpuCount === 1) {
    return { strategy: 'auto', gpu_count: 1 };
  }
  const strategy = values.resource_strategy === 'fixed' ? 'fixed' : 'auto';
  return {
    strategy,
    gpu_count: positive(values.gpu_count) ? Number(values.gpu_count) : 1,
    ...(strategy === 'fixed' ? { gpu_ids: [...(values.gpu_ids ?? [])] } : {}),
  };
}

export function capabilityMatches(input: string, values: Array<string | undefined>): boolean {
  const term = input.trim().toLocaleLowerCase();
  if (!term) return true;
  return values.filter(Boolean).join(' ').toLocaleLowerCase().includes(term);
}

export function filterSupportedWandbCategories(selected: string[] | undefined, supported: string[]): string[] {
  const allowed = new Set(supported);
  return Array.from(new Set(selected ?? [])).filter((id) => allowed.has(id));
}
