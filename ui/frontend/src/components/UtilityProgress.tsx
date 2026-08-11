import { Progress, Space, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import type { UtilityRun } from '../api/types';

function amount(value?: number, unit?: string): string | undefined {
  if (value == null) return undefined;
  if (unit !== 'bytes') return value.toLocaleString();
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

export function UtilityProgress({ run, compact = false }: { run: UtilityRun; compact?: boolean }) {
  const { t } = useTranslation();
  const progress = run.progress;
  const percent = run.status === 'completed' ? 100 : Math.round(progress?.percent ?? 0);
  const failed = ['failed', 'interrupted'].includes(run.status);
  const stopped = ['stopped', 'cancelled'].includes(run.status);
  const phase = progress?.phase
    ? t(`utility.phase.${progress.phase}`, { defaultValue: progress.phase })
    : t(`status.${run.status}`, { defaultValue: run.status });
  const current = amount(progress?.current, progress?.unit);
  const total = amount(progress?.total, progress?.unit);
  return (
    <Space direction="vertical" size={compact ? 0 : 2} style={{ width: compact ? 150 : '100%' }}>
      <Progress
        percent={percent}
        size="small"
        status={failed ? 'exception' : run.status === 'completed' ? 'success' : stopped ? 'normal' : 'active'}
        strokeColor={stopped ? '#b8860b' : undefined}
      />
      <Typography.Text type="secondary" style={{ fontSize: 11 }}>
        {phase}{current && total ? ` · ${current} / ${total}` : ''}
      </Typography.Text>
      {failed && (progress?.message || run.error) ? (
        <Typography.Text
          type="danger"
          style={{ maxWidth: '100%', fontSize: 11, whiteSpace: 'normal', overflowWrap: 'anywhere' }}
        >
          {progress?.message || run.error}
        </Typography.Text>
      ) : null}
    </Space>
  );
}
