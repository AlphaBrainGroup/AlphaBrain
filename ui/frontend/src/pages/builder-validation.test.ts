import { describe, expect, it } from 'vitest';
import { builderGpuIds, capabilityMatches, defaultWandbRunConfig, filterSupportedWandbCategories, isBuilderStepComplete, normalizeBuilderResources, workflowFieldActive } from './builder-validation';

const options = {
  localDatasetValid: false,
  requiresCheckpoint: false,
  requiredParameterKeys: [] as string[],
  expertYamlValid: true,
};

describe('experiment builder gating', () => {
  it('starts with W&B disabled and safe non-secret content defaults', () => {
    expect(defaultWandbRunConfig()).toEqual({
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
    });
    expect(defaultWandbRunConfig()).not.toHaveProperty('api_key');
  });

  it('keeps architecture incomplete until every required selection is made', () => {
    const selected = { backbone_id: 'qwen', action_head_id: 'mlp', method_id: 'il', dataset_id: 'libero' };
    expect(isBuilderStepComplete(0, { ...selected, dataset_id: undefined }, options)).toBe(false);
    expect(isBuilderStepComplete(0, selected, options)).toBe(true);
    expect(isBuilderStepComplete(0, { ...selected, use_local_dataset: true, dataset_path: '/data' }, options)).toBe(false);
    expect(isBuilderStepComplete(0, { ...selected, use_local_dataset: true, dataset_path: '/data' }, { ...options, localDatasetValid: true })).toBe(true);
    expect(isBuilderStepComplete(0, { ...selected, dataset_source: 'registered', dataset_registration_id: 'dataset-1' }, { ...options, datasetReferenceReady: true })).toBe(true);
    expect(isBuilderStepComplete(0, { ...selected, dataset_source: 'mixture' }, { ...options, datasetReferenceReady: false })).toBe(false);
  });

  it('requires all parameter-step values and conditional fields', () => {
    const values = { name: 'run', batch_size: 4, learning_rate: 0.0001, max_steps: 100, save_interval: 10, resource_strategy: 'auto' as const, gpu_count: 1 };
    expect(isBuilderStepComplete(1, values, options)).toBe(true);
    expect(isBuilderStepComplete(1, { ...values, learning_rate: 0 }, options)).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, resource_strategy: 'fixed', gpu_ids: [] }, options)).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, resource_strategy: 'fixed', gpu_count: 2, gpu_ids: [0] }, options)).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, resource_strategy: 'fixed', gpu_count: 2, gpu_ids: [0, 1] }, options)).toBe(true);
    expect(isBuilderStepComplete(1, values, { ...options, requiresCheckpoint: true })).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, resume_mode: 'full_state' }, options)).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, resume_mode: 'full_state', resume_checkpoint: '/runs/steps_10' }, options)).toBe(true);
    expect(isBuilderStepComplete(1, values, { ...options, requiredWorkflowKeys: ['encoder_checkpoint'] })).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, workflow_config: { encoder_checkpoint: '/models/encoder.pt' } }, { ...options, requiredWorkflowKeys: ['encoder_checkpoint'] })).toBe(true);
  });

  it('gates enabled W&B config and rejects unsupported upload categories', () => {
    const values = { name: 'run', batch_size: 4, learning_rate: 0.0001, max_steps: 100, save_interval: 10, resource_strategy: 'auto' as const, gpu_count: 1 };
    const supported = { ...options, supportedWandbCategoryIds: ['metrics', 'config', 'system'] };
    expect(isBuilderStepComplete(1, { ...values, wandb: { enabled: false } }, supported)).toBe(true);
    expect(isBuilderStepComplete(1, { ...values, wandb: { enabled: true, mode: 'online', project: '', categories: ['metrics'] } }, supported)).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, wandb: { enabled: true, mode: 'disabled', project: 'AlphaBrain', categories: ['metrics'] } }, supported)).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, wandb: { enabled: true, mode: 'offline', project: 'AlphaBrain', categories: ['videos'] } }, supported)).toBe(false);
    expect(isBuilderStepComplete(1, { ...values, wandb: { enabled: true, mode: 'offline', project: 'AlphaBrain', categories: ['metrics', 'config'] } }, supported)).toBe(true);
    expect(filterSupportedWandbCategories(['metrics', 'videos', 'metrics', 'checkpoints'], ['metrics', 'config', 'system'])).toEqual(['metrics']);
  });

  it('does not expose invalid resource state on a single-GPU host', () => {
    const values = { name: 'run', batch_size: 4, learning_rate: 0.0001, max_steps: 100, save_interval: 10 };
    expect(isBuilderStepComplete(1, values, { ...options, singleGpu: true })).toBe(true);
    expect(normalizeBuilderResources({ resource_strategy: 'fixed', gpu_count: 8, gpu_ids: [3] }, 1)).toEqual({
      strategy: 'auto',
      gpu_count: 1,
    });
  });

  it('preserves automatic and fixed allocation on multi-GPU or undetected hosts', () => {
    expect(normalizeBuilderResources({ resource_strategy: 'auto', gpu_count: 2, gpu_ids: [0] }, 4)).toEqual({
      strategy: 'auto',
      gpu_count: 2,
    });
    expect(normalizeBuilderResources({ resource_strategy: 'fixed', gpu_count: 2, gpu_ids: [1, 3] }, 4)).toEqual({
      strategy: 'fixed',
      gpu_count: 2,
      gpu_ids: [1, 3],
    });
  });

  it('uses configured SSH server GPUs when remote training is enabled', () => {
    expect(builderGpuIds([0, 1], { mode: 'remote', gpu_ids: [0, 1, 2, 3] })).toEqual([0, 1, 2, 3]);
    expect(builderGpuIds([0, 1], { mode: 'local', gpu_ids: [] })).toEqual([0, 1]);
    expect(builderGpuIds([0, 1], { mode: 'remote', gpu_ids: [3, 3, -1, 1] })).toEqual([3, 1]);
  });
});

describe('workflow field visibility', () => {
  it('honors value conditions and registry method/algorithm scopes', () => {
    const reuse = { key: 'encoder_checkpoint', type: 'path' as const, visible_when: { key: 'encoder_mode', equals: 'reuse' } };
    expect(workflowFieldActive(reuse, { encoder_mode: 'train' })).toBe(false);
    expect(workflowFieldActive(reuse, { encoder_mode: 'reuse' })).toBe(true);
    expect(workflowFieldActive({ key: 'tau', type: 'number', algorithms: ['td3'] }, {}, { algorithm: 'grpo' })).toBe(false);
    expect(workflowFieldActive({ key: 'tau', type: 'number', algorithms: ['td3'] }, {}, { algorithm: 'td3' })).toBe(true);
    expect(workflowFieldActive({ key: 'ewc_lambda', type: 'number', methods: ['continual_ewc'] }, {}, { method: 'continual_mir' })).toBe(false);
  });
});

describe('capability search', () => {
  it('matches ids and localized model names case-insensitively', () => {
    expect(capabilityMatches('Q', ['qwen2_5_vl', 'Qwen 2.5 VL', '千问视觉模型'])).toBe(true);
    expect(capabilityMatches('千问', ['qwen2_5_vl', 'Qwen 2.5 VL', '千问视觉模型'])).toBe(true);
    expect(capabilityMatches('cosmos', ['qwen2_5_vl', 'Qwen 2.5 VL'])).toBe(false);
  });
});
