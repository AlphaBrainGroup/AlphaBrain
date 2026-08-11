import type { DeploymentCreateRequest, ModelDeployment, ParameterSchema } from '../api/types';

export function normalizeDeploymentResources(
  resources: Partial<DeploymentCreateRequest['resources']> | undefined,
  visibleGpuCount?: number,
): DeploymentCreateRequest['resources'] {
  if (visibleGpuCount === 1) {
    return { strategy: 'auto', gpu_count: 1, gpu_ids: [] };
  }
  const strategy = resources?.strategy === 'fixed' ? 'fixed' : 'auto';
  const requestedCount = Number(resources?.gpu_count);
  const gpuCount = Number.isInteger(requestedCount) && requestedCount >= 1 ? requestedCount : 1;
  return {
    strategy,
    gpu_count: gpuCount,
    ...(strategy === 'fixed' ? { gpu_ids: (resources?.gpu_ids ?? []).map(Number) } : {}),
  };
}

export function mergeParameterSchemas(...groups: Array<ParameterSchema[] | undefined>): ParameterSchema[] {
  const merged = new Map<string, ParameterSchema>();
  groups.forEach((group) => group?.forEach((parameter) => {
    merged.set(parameter.key, { ...merged.get(parameter.key), ...parameter });
  }));
  return Array.from(merged.values());
}

export function parameterDefaults(parameters: ParameterSchema[]): Record<string, unknown> {
  return Object.fromEntries(parameters
    .filter((parameter) => parameter.default !== undefined)
    .map((parameter) => [parameter.key, parameter.default]));
}

export function deploymentEndpoint(deployment: ModelDeployment): string | undefined {
  if (deployment.endpoint.url) return deployment.endpoint.url;
  if (!deployment.endpoint.port) return undefined;
  const host = deployment.endpoint.advertised_host
    ?? (deployment.endpoint.scope === 'local' ? '127.0.0.1' : undefined);
  return host ? `ws://${host}:${deployment.endpoint.port}` : undefined;
}

export function pythonClientSnippet(endpoint: string, apiKey: string): string {
  return `from websocket import create_connection

ws = create_connection(
    ${JSON.stringify(endpoint)},
    header=[${JSON.stringify(`Authorization: Api-Key ${apiKey}`)}],
)
# Send observations using the adapter protocol documented by AlphaBrain.
`;
}
