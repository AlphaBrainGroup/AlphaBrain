/** @vitest-environment jsdom */

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import '../i18n';
import { GpuCard } from './GpuCard';

afterEach(cleanup);

describe('GpuCard', () => {
  it('renders probe failures without misleading utilization or occupancy', () => {
    render(<GpuCard gpu={{
      index: 0,
      name: 'GPU 0',
      memory_used_mb: 0,
      memory_total_mb: 0,
      utilization_percent: 0,
      available: false,
      error: 'memory: device lost',
    }} />);

    expect(screen.getAllByText('检测失败')).toHaveLength(2);
    expect(screen.getByText('memory: device lost')).not.toBeNull();
    expect(screen.queryByRole('progressbar')).toBeNull();
    expect(screen.queryByText('使用中')).toBeNull();
  });

  it('keeps normal metrics for a healthy GPU', () => {
    render(<GpuCard gpu={{
      index: 0,
      name: 'RTX 4090',
      memory_used_mb: 6144,
      memory_total_mb: 24576,
      utilization_percent: 42,
      temperature_c: 55,
      available: true,
    }} />);

    expect(screen.getByText('RTX 4090')).not.toBeNull();
    expect(screen.getByText('空闲')).not.toBeNull();
    expect(screen.getByRole('progressbar')).not.toBeNull();
  });
});
