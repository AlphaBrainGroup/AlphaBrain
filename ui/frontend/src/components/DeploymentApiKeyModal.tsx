import { CopyOutlined } from '@ant-design/icons';
import { Alert, Button, Input, Modal, Space, Typography, message } from 'antd';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { ModelDeployment } from '../api/types';
import { deploymentEndpoint, pythonClientSnippet } from '../pages/deployment-form';

interface Props {
  deployment?: ModelDeployment;
  apiKey?: string;
  open: boolean;
  onClose: () => void;
}

export function DeploymentApiKeyModal({ deployment, apiKey, open, onClose }: Props) {
  const { t } = useTranslation();
  const endpoint = deployment ? deploymentEndpoint(deployment) : undefined;
  const snippet = useMemo(
    () => apiKey ? pythonClientSnippet(endpoint ?? 'ws://HOST:PORT', apiKey) : '',
    [apiKey, endpoint],
  );
  const copy = (value: string) => {
    void navigator.clipboard.writeText(value);
    message.success(t('common.copied'));
  };

  return (
    <Modal
      open={open}
      width={720}
      title={t('deployment.apiKeyTitle')}
      okText={t('deployment.savedKey')}
      cancelButtonProps={{ style: { display: 'none' } }}
      closable={false}
      maskClosable={false}
      keyboard={false}
      onOk={onClose}
    >
      <Alert type="warning" showIcon message={t('deployment.apiKeyOnce')} description={t('deployment.apiKeyOnceDesc')} />
      <Typography.Title level={5}>{t('deployment.apiKey')}</Typography.Title>
      <Space.Compact block>
        <Input.Password readOnly visibilityToggle value={apiKey} />
        <Button icon={<CopyOutlined />} onClick={() => apiKey && copy(apiKey)}>{t('common.copy')}</Button>
      </Space.Compact>
      <Typography.Title level={5}>{t('deployment.clientExample')}</Typography.Title>
      <pre className="command-preview deployment-snippet">{snippet}</pre>
      <Button icon={<CopyOutlined />} onClick={() => copy(snippet)}>{t('deployment.copyExample')}</Button>
    </Modal>
  );
}
