import { FolderOpenOutlined, FolderOutlined, ReloadOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Button, Input, List, Modal, Space } from 'antd';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api } from '../api/client';

interface Props {
  value?: string;
  placeholder?: string;
  onChange?: (value: string) => void;
}

export function ServerDirectoryPicker({ value, placeholder, onChange }: Props) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [browsePath, setBrowsePath] = useState<string | undefined>(value || undefined);
  const [address, setAddress] = useState(value ?? '');
  const directories = useQuery({
    queryKey: ['server-directories', browsePath ?? 'home'],
    queryFn: () => api.datasets.directories(browsePath),
    enabled: open,
    retry: false,
  });

  useEffect(() => {
    if (directories.data?.path) setAddress(directories.data.path);
  }, [directories.data?.path]);

  return (
    <>
      <Space.Compact block>
        <Input
          value={value}
          placeholder={placeholder}
          onChange={(event) => onChange?.(event.target.value)}
        />
        <Button icon={<FolderOpenOutlined />} onClick={() => { setBrowsePath(value || undefined); setOpen(true); }}>
          {t('builder.browseFolder')}
        </Button>
      </Space.Compact>
      <Modal
        open={open}
        width={720}
        title={t('deployment.chooseCheckpointFolder')}
        okText={t('builder.chooseThisFolder')}
        okButtonProps={{ disabled: !directories.data?.path }}
        onOk={() => {
          const selected = directories.data?.path ?? address;
          if (!selected) return;
          onChange?.(selected);
          setOpen(false);
        }}
        onCancel={() => setOpen(false)}
      >
        <Space.Compact block className="directory-address">
          <Input value={address} onChange={(event) => setAddress(event.target.value)} onPressEnter={() => setBrowsePath(address)} />
          <Button icon={<ReloadOutlined />} onClick={() => setBrowsePath(address)}>{t('builder.goToFolder')}</Button>
        </Space.Compact>
        <List
          className="directory-list"
          loading={directories.isLoading}
          locale={{
            emptyText: directories.error
              ? (directories.error instanceof Error ? directories.error.message : String(directories.error))
              : t('builder.noSubdirectories'),
          }}
          dataSource={[
            ...(directories.data?.parent ? [{ name: '..', path: directories.data.parent }] : []),
            ...(directories.data?.directories ?? []),
          ]}
          renderItem={(item) => (
            <List.Item className="directory-item" onClick={() => setBrowsePath(item.path)}>
              <Space><FolderOutlined /><span>{item.name}</span></Space>
            </List.Item>
          )}
        />
      </Modal>
    </>
  );
}
