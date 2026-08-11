import { describe, expect, it } from 'vitest';
import type { ModelDeployment, ParameterSchema } from '../api/types';
import {
  deploymentEndpoint,
  mergeParameterSchemas,
  normalizeDeploymentResources,
  parameterDefaults,
  pythonClientSnippet,
} from './deployment-form';

describe('deployment form helpers', () => {
  it('merges adapter and combination schemas by key with combination overrides', () => {
    const adapter: ParameterSchema[] = [
      { key: 'precision', type: 'select', default: 'bf16', options: [{ label: 'BF16', value: 'bf16' }] },
      { key: 'trust_remote_code', type: 'boolean', default: false },
    ];
    const combination: ParameterSchema[] = [
      { key: 'precision', type: 'select', default: 'fp16', required: true, options: [{ label: 'FP16', value: 'fp16' }] },
    ];
    const merged = mergeParameterSchemas(adapter, combination);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toMatchObject({ key: 'precision', default: 'fp16', required: true });
    expect(parameterDefaults(merged)).toEqual({ precision: 'fp16', trust_remote_code: false });
  });

  it('builds endpoint and a Python client without losing the API key', () => {
    const deployment = {
      endpoint: { scope: 'lan', advertised_host: 'lab-gpu.local', port: 8765 },
    } as ModelDeployment;
    expect(deploymentEndpoint(deployment)).toBe('ws://lab-gpu.local:8765');
    expect(pythonClientSnippet('ws://lab-gpu.local:8765', 'secret-key')).toContain('Authorization: Api-Key secret-key');
  });

  it('forces stale fixed or template resources to the only visible GPU', () => {
    expect(normalizeDeploymentResources({ strategy: 'fixed', gpu_count: 4, gpu_ids: [3] }, 1)).toEqual({
      strategy: 'auto',
      gpu_count: 1,
      gpu_ids: [],
    });
  });

  it('preserves valid multi-GPU resource choices', () => {
    expect(normalizeDeploymentResources({ strategy: 'fixed', gpu_count: 2, gpu_ids: [1, 3] }, 4)).toEqual({
      strategy: 'fixed',
      gpu_count: 2,
      gpu_ids: [1, 3],
    });
    expect(normalizeDeploymentResources({ strategy: 'auto', gpu_count: 2 }, 4)).toEqual({
      strategy: 'auto',
      gpu_count: 2,
    });
  });
});
