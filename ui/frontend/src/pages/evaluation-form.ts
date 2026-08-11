import type {
  EvaluationBenchmark,
  EvaluationCreateRequest,
  EvaluationPreset,
  EvaluationRun,
  ParameterSchema,
} from '../api/types';

export function normalizeEvaluationResources(
  resources: Partial<EvaluationCreateRequest['resources']> | undefined,
  visibleGpuCount?: number,
  allowMultiple = false,
): EvaluationCreateRequest['resources'] {
  const requestedCount = allowMultiple ? Math.max(1, Math.trunc(Number(resources?.gpu_count ?? 1))) : 1;
  if (visibleGpuCount === 1) return { strategy: 'auto', gpu_count: 1, gpu_ids: [] };
  if (resources?.strategy === 'fixed') {
    const gpuIds = (resources.gpu_ids ?? []).slice(0, requestedCount).map(Number);
    return { strategy: 'fixed', gpu_count: requestedCount, gpu_ids: gpuIds };
  }
  return { strategy: 'auto', gpu_count: requestedCount };
}

export function evaluationParameterDefaults(benchmark?: EvaluationBenchmark): Record<string, unknown> {
  if (!benchmark) return {};
  const schemaDefaults = Object.fromEntries(benchmark.parameters
    .filter((parameter) => parameter.default !== undefined)
    .map((parameter) => [parameter.key, parameter.default]));
  return { ...benchmark.defaults, ...schemaDefaults };
}

export function presetParameters(benchmark: EvaluationBenchmark | undefined, preset: EvaluationPreset): Record<string, unknown> {
  if (!benchmark) return {};
  // The backend expands quick/standard/full into benchmark-specific field
  // names. Sending schema defaults here would override that expansion (for
  // example, standard LIBERO would be forced back to one trial).
  return preset === 'custom' ? evaluationParameterDefaults(benchmark) : {};
}

export function parameterHasValue(parameters: Record<string, unknown> | undefined, schema: ParameterSchema): boolean {
  if (!schema.required || schema.type === 'boolean') return true;
  const value = parameters?.[schema.key];
  return value !== undefined && value !== null && value !== '';
}

function stringParameter(parameters: Record<string, unknown>, key: string): string {
  const value = parameters[key];
  return value == null ? '' : String(value).trim();
}

function integerParameter(parameters: Record<string, unknown>, key: string): number {
  const value = Number(parameters[key] ?? 0);
  return Number.isFinite(value) ? Math.trunc(value) : 0;
}

export function evaluationSignature(run: EvaluationRun): string {
  const parameters = run.parameters ?? {};
  return JSON.stringify([
    run.benchmark_id,
    run.suite ?? '',
    run.task_set ?? '',
    run.split ?? '',
    stringParameter(parameters, 'task_ids'),
    stringParameter(parameters, 'task_list'),
    integerParameter(parameters, 'task_limit'),
  ]);
}

export function comparisonIssue(runs: EvaluationRun[]): 'count' | 'status' | 'benchmark' | 'signature' | undefined {
  if (runs.length < 2 || runs.length > 8) return 'count';
  if (runs.some((run) => run.status !== 'completed')) return 'status';
  if (new Set(runs.map((run) => run.benchmark_id)).size > 1) return 'benchmark';
  if (new Set(runs.map(evaluationSignature)).size > 1) return 'signature';
  return undefined;
}
