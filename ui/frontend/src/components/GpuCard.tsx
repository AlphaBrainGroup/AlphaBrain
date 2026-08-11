import { ApiOutlined } from '@ant-design/icons';
import { Alert, Card, Progress, Space, Tag, Tooltip, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import type { GPU } from '../api/types';

function formatMemory(mb: number): string {
  return `${(mb / 1024).toFixed(1)} GB`;
}

export function GpuCard({ gpu }: { gpu: GPU }) {
  const { t } = useTranslation();
  const memoryPercent = gpu.memory_total_mb > 0 ? Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100) : 0;
  const external = Boolean(gpu.external_processes && !gpu.job);
  const failed = Boolean(gpu.error);
  const available = !failed && (gpu.available ?? (!gpu.job && !external));
  return (
    <Card className="gpu-card" size="small">
      <div className="gpu-card-head">
        <Space>
          <span className={`gpu-index ${available ? 'free' : ''}`}>{gpu.index}</span>
          <div>
            <Typography.Text strong>GPU {gpu.index}</Typography.Text>
            <Typography.Text type="secondary" className="gpu-name">{gpu.name}</Typography.Text>
          </div>
        </Space>
        <Tag color={failed ? 'red' : available ? 'green' : external ? 'orange' : 'blue'}>
          {failed ? t('dashboard.probeFailed') : available ? t('dashboard.free') : external ? t('dashboard.external') : t('dashboard.occupied')}
        </Tag>
      </div>
      {failed ? (
        <Alert type="error" showIcon message={t('dashboard.probeFailed')} description={gpu.error} />
      ) : (
        <>
          <Tooltip title={`${formatMemory(gpu.memory_used_mb)} / ${formatMemory(gpu.memory_total_mb)}`}>
            <Progress percent={memoryPercent} size="small" strokeColor={memoryPercent > 90 ? '#e5484d' : '#3157d5'} />
          </Tooltip>
          <div className="gpu-meta">
            <span><ApiOutlined /> {gpu.utilization_percent}%</span>
            {gpu.temperature_c != null ? <span>{gpu.temperature_c} °C</span> : null}
            {gpu.job ? <span className="ellipsis">{gpu.job.name}</span> : null}
          </div>
        </>
      )}
    </Card>
  );
}
