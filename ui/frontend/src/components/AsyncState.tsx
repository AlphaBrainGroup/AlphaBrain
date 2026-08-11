import type { ReactNode } from 'react';
import { Alert, Button, Empty, Skeleton, Space } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

interface AsyncStateProps {
  loading?: boolean;
  error?: unknown;
  empty?: boolean;
  onRetry?: () => void;
  children: ReactNode;
}

export function AsyncState({ loading, error, empty, onRetry, children }: AsyncStateProps) {
  const { t } = useTranslation();
  if (loading) return <Skeleton active paragraph={{ rows: 6 }} />;
  if (error) {
    return (
      <Alert
        type="error"
        showIcon
        message={t('common.networkError')}
        description={
          <Space direction="vertical">
            <span>{error instanceof Error ? error.message : t('common.networkHint')}</span>
            {onRetry ? <Button icon={<ReloadOutlined />} onClick={onRetry}>{t('common.retry')}</Button> : null}
          </Space>
        }
      />
    );
  }
  if (empty) return <Empty description={t('common.noData')} />;
  return <>{children}</>;
}
