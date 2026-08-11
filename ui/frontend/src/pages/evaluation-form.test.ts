import { describe, expect, it } from 'vitest';
import type { EvaluationBenchmark, EvaluationRun } from '../api/types';
import {
  comparisonIssue,
  evaluationParameterDefaults,
  normalizeEvaluationResources,
  presetParameters,
} from './evaluation-form';

const benchmark = {
  id: 'libero',
  parameters: [
    { key: 'seed', type: 'integer', default: 7 },
    { key: 'episodes_per_task', type: 'integer', default: 10 },
  ],
  defaults: { seed: 1, max_steps: 600 },
  presets: [{ id: 'quick', name: 'Quick', parameters: { episodes_per_task: 1 } }],
} as unknown as EvaluationBenchmark;

describe('evaluation form helpers', () => {
  it('always requests one GPU and hides a stale fixed choice on a single-GPU host', () => {
    expect(normalizeEvaluationResources({ strategy: 'fixed', gpu_count: 1, gpu_ids: [3] }, 1)).toEqual({
      strategy: 'auto', gpu_count: 1, gpu_ids: [],
    });
    expect(normalizeEvaluationResources({ strategy: 'fixed', gpu_count: 1, gpu_ids: [2] }, 4)).toEqual({
      strategy: 'fixed', gpu_count: 1, gpu_ids: [2],
    });
    expect(normalizeEvaluationResources({ strategy: 'fixed', gpu_count: 2, gpu_ids: [2, 3] }, 4, true)).toEqual({
      strategy: 'fixed', gpu_count: 2, gpu_ids: [2, 3],
    });
  });

  it('only sends explicit benchmark parameters for the custom preset', () => {
    expect(evaluationParameterDefaults(benchmark)).toEqual({ seed: 7, max_steps: 600, episodes_per_task: 10 });
    expect(presetParameters(benchmark, 'quick')).toEqual({});
    expect(presetParameters(benchmark, 'custom')).toEqual({ seed: 7, max_steps: 600, episodes_per_task: 10 });
  });

  it('only compares runs with the same benchmark and task signature', () => {
    const run: EvaluationRun = {
      id: 'one',
      name: 'First evaluation',
      status: 'completed',
      combination_id: 'qwen-oft',
      benchmark_id: 'libero',
      preset: 'custom',
      suite: 'goal',
      task_set: undefined,
      split: 'test',
      parameters: { task_ids: ' 1,2 ', task_list: ' task-a\ntask-b ', task_limit: 2 },
      wandb: { enabled: false, mode: 'offline', project: '', entity: '', run_name: '', tags: [], categories: [] },
      requested_gpu_ids: [],
      assigned_gpu_ids: [],
      videos: [],
    };
    expect(comparisonIssue([run])).toBe('count');
    expect(comparisonIssue([run, { ...run, id: 'two' }])).toBeUndefined();
    expect(comparisonIssue([run, {
      ...run,
      id: 'two',
      parameters: { task_ids: '1,2', task_list: 'task-a\ntask-b', task_limit: 2 },
    }])).toBeUndefined();
    expect(comparisonIssue([run, { ...run, id: 'two', status: 'running' }])).toBe('status');
    expect(comparisonIssue([run, { ...run, id: 'two', benchmark_id: 'robocasa365' }])).toBe('benchmark');
    expect(comparisonIssue([run, { ...run, id: 'two', suite: 'spatial' }])).toBe('signature');
    expect(comparisonIssue([run, { ...run, id: 'two', parameters: { ...run.parameters, task_ids: '1,3' } }])).toBe('signature');
    expect(comparisonIssue([run, { ...run, id: 'two', parameters: { ...run.parameters, task_list: 'task-a' } }])).toBe('signature');
    expect(comparisonIssue([run, { ...run, id: 'two', parameters: { ...run.parameters, task_limit: 1 } }])).toBe('signature');
  });
});
