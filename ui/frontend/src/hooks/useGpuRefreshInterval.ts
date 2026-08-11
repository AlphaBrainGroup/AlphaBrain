import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';

export const DEFAULT_GPU_REFRESH_INTERVAL_SECONDS = 5;

export function normalizeGpuRefreshIntervalSeconds(value: unknown): number {
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return DEFAULT_GPU_REFRESH_INTERVAL_SECONDS;
  return Math.min(3600, Math.round(parsed));
}

/** React Query interval in milliseconds; false means manual refresh only. */
export function useGpuRefreshInterval(): number | false {
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me, staleTime: 60_000 });
  const seconds = normalizeGpuRefreshIntervalSeconds(me.data?.gpu_refresh_interval_seconds);
  return seconds === 0 ? false : seconds * 1000;
}
