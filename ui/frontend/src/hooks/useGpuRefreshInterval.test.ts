import { describe, expect, it } from 'vitest';
import { DEFAULT_GPU_REFRESH_INTERVAL_SECONDS, normalizeGpuRefreshIntervalSeconds } from './useGpuRefreshInterval';

describe('GPU refresh interval preference', () => {
  it('keeps manual-only and supported positive intervals', () => {
    expect(normalizeGpuRefreshIntervalSeconds(0)).toBe(0);
    expect(normalizeGpuRefreshIntervalSeconds(2)).toBe(2);
    expect(normalizeGpuRefreshIntervalSeconds(30)).toBe(30);
  });

  it('falls back for invalid values and caps excessive values', () => {
    expect(normalizeGpuRefreshIntervalSeconds(undefined)).toBe(DEFAULT_GPU_REFRESH_INTERVAL_SECONDS);
    expect(normalizeGpuRefreshIntervalSeconds(-1)).toBe(DEFAULT_GPU_REFRESH_INTERVAL_SECONDS);
    expect(normalizeGpuRefreshIntervalSeconds(Number.NaN)).toBe(DEFAULT_GPU_REFRESH_INTERVAL_SECONDS);
    expect(normalizeGpuRefreshIntervalSeconds(9999)).toBe(3600);
  });
});
