import { Tag } from 'antd';
import { useTranslation } from 'react-i18next';
import type { EvaluationStatus, JobStatus } from '../api/types';

const colors: Record<string, string> = {
  draft: 'default',
  checking: 'processing',
  blocked: 'purple',
  queued: 'gold',
  starting: 'cyan',
  running: 'blue',
  stopping: 'orange',
  completed: 'green',
  stopped: 'default',
  failed: 'red',
  interrupted: 'volcano',
  dependency_failed: 'red',
  cancelled: 'default',
};

export function StatusTag({ status }: { status?: JobStatus | EvaluationStatus | string }) {
  const { t } = useTranslation();
  if (!status) return <Tag>{t('common.unknown')}</Tag>;
  return <Tag color={colors[status] ?? 'default'}>{t(`status.${status}`)}</Tag>;
}
