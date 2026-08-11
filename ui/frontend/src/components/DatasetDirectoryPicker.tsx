import { CheckCircleFilled, FolderOpenOutlined, FolderOutlined, ReloadOutlined, WarningFilled } from '@ant-design/icons';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Alert, Button, Input, List, Modal, Space, Typography, message } from 'antd';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api } from '../api/client';
import type { DatasetValidationResult, Language } from '../api/types';

interface Props {
  value?: string;
  datasetId?: string;
  language: Language;
  onChange?: (value: string) => void;
  onValidated: (result?: DatasetValidationResult) => void;
  validation?: DatasetValidationResult;
}

export function DatasetDirectoryPicker({ value, datasetId, language, onChange, onValidated, validation }: Props) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [browsePath, setBrowsePath] = useState<string | undefined>(value || undefined);
  const [address, setAddress] = useState(value ?? '');
  const directories = useQuery({
    queryKey: ['dataset-directories', browsePath ?? 'home'],
    queryFn: () => api.datasets.directories(browsePath),
    enabled: open,
    retry: false,
  });
  useEffect(() => {
    if (directories.data?.path) setAddress(directories.data.path);
  }, [directories.data?.path]);

  const validate = useMutation({
    mutationFn: async (path: string) => {
      if (!datasetId) throw new Error(t('builder.selectDatasetFirst'));
      return api.datasets.validate(path, datasetId);
    },
    onSuccess: (result) => {
      if (result.valid) {
        onChange?.(result.normalized_root);
        message.success(t('builder.datasetValid'));
      }
      onValidated(result);
    },
    onError: (error) => {
      onValidated(undefined);
      message.error(error instanceof Error ? error.message : String(error));
    },
  });

  const chooseCurrent = () => {
    const selected = directories.data?.path ?? address;
    if (!selected) return;
    onChange?.(selected);
    onValidated(undefined);
    setOpen(false);
    validate.mutate(selected);
  };
  const issueText = validation?.issues
    .map((item) => item.message_i18n?.[language] ?? item.message ?? item.title)
    .join('\n');

  return (
    <div className="dataset-picker">
      <Space.Compact block>
        <Input
          value={value}
          placeholder={t('builder.datasetPathPlaceholder')}
          onChange={(event) => { onChange?.(event.target.value); onValidated(undefined); }}
          onPressEnter={() => value && validate.mutate(value)}
        />
        <Button icon={<FolderOpenOutlined />} onClick={() => { setBrowsePath(value || undefined); setOpen(true); }}>
          {t('builder.browseFolder')}
        </Button>
        <Button loading={validate.isPending} disabled={!value || !datasetId} onClick={() => value && validate.mutate(value)}>
          {t('builder.validateDataset')}
        </Button>
      </Space.Compact>
      {validation ? (
        <Alert
          className="dataset-validation"
          type={validation.valid ? 'success' : 'error'}
          showIcon
          icon={validation.valid ? <CheckCircleFilled /> : <WarningFilled />}
          message={validation.valid
            ? t('builder.datasetValidationSummary', { format: validation.format, datasets: validation.dataset_count, episodes: validation.episode_count, parquet: validation.parquet_count })
            : t('builder.datasetInvalid')}
          description={issueText ? <Typography.Paragraph className="dataset-validation-detail">{issueText}</Typography.Paragraph> : undefined}
        />
      ) : null}
      <Modal
        open={open}
        width={720}
        title={t('builder.chooseDatasetFolder')}
        okText={t('builder.chooseThisFolder')}
        okButtonProps={{ disabled: !directories.data?.path }}
        onOk={chooseCurrent}
        onCancel={() => setOpen(false)}
      >
        <Space.Compact block className="directory-address">
          <Input value={address} onChange={(event) => setAddress(event.target.value)} onPressEnter={() => setBrowsePath(address)} />
          <Button icon={<ReloadOutlined />} onClick={() => setBrowsePath(address)}>{t('builder.goToFolder')}</Button>
        </Space.Compact>
        <List
          className="directory-list"
          loading={directories.isLoading}
          locale={{ emptyText: directories.error ? (directories.error instanceof Error ? directories.error.message : String(directories.error)) : t('builder.noSubdirectories') }}
          dataSource={[
            ...(directories.data?.parent ? [{ name: '..', path: directories.data.parent }] : []),
            ...(directories.data?.directories ?? []),
          ]}
          renderItem={(item) => (
            <List.Item className="directory-item" onClick={() => setBrowsePath(item.path)}>
              <Space><FolderOutlined /> <span>{item.name}</span></Space>
            </List.Item>
          )}
        />
      </Modal>
    </div>
  );
}
