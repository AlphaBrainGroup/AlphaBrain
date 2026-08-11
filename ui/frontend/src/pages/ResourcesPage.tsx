import { CloudDownloadOutlined, FolderOpenOutlined, KeyOutlined, PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Form, Input, InputNumber, Modal, Select, Space, Tag, Typography, message } from 'antd';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api } from '../api/client';
import type { ResourceRecord } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { UtilityProgress } from '../components/UtilityProgress';

type ResourceAction = { kind: 'install' | 'register' | 'preprocess'; resource: ResourceRecord };

export function ResourcesPage() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const resources = useQuery({ queryKey: ['resources'], queryFn: api.resources.list, refetchInterval: 5_000 });
  const utilities = useQuery({ queryKey: ['utilities'], queryFn: api.utilities.list, refetchInterval: 3_000 });
  const globalToken = useQuery({ queryKey: ['hf-download-token'], queryFn: api.settings.huggingface.downloadToken.status, enabled: me.data?.role === 'administrator' });
  const publishToken = useQuery({ queryKey: ['hf-publish-token'], queryFn: api.settings.huggingface.publishToken.status });
  const [action, setAction] = useState<ResourceAction>();
  const [form] = Form.useForm();
  const [globalValue, setGlobalValue] = useState('');
  const [publishValue, setPublishValue] = useState('');

  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['resources'] }),
      queryClient.invalidateQueries({ queryKey: ['utilities'] }),
    ]);
  };
  const submitAction = useMutation({
    mutationFn: async (values: Record<string, unknown>) => {
      if (!action) throw new Error('missing_resource_action');
      if (action.kind === 'install') return api.resources.install(action.resource.id, String(values.target_root ?? ''));
      if (action.kind === 'register') return api.resources.registerPath(action.resource.id, String(values.path ?? ''));
      return api.resources.preprocess(action.resource.id, {
        kind: String(values.preprocess_kind),
        inputs: {
          data_root: String(values.data_root ?? ''),
          output_path: String(values.output_path ?? ''),
          src: String(values.source_path ?? ''),
          dst: String(values.output_path ?? ''),
        },
        gpu_count: Number(values.gpu_count ?? 1),
      });
    },
    onSuccess: async () => { message.success(t('resources.taskCreated')); setAction(undefined); form.resetFields(); await refresh(); },
    onError: (error) => message.error(error.message),
  });
  const cancelRun = useMutation({
    mutationFn: (id: string) => api.utilities.cancel(id),
    onSuccess: async () => {
      message.success(t('resources.taskCancelled'));
      await refresh();
    },
    onError: (error) => message.error(error.message),
  });
  const saveToken = async (scope: 'global' | 'publish') => {
    const value = (scope === 'global' ? globalValue : publishValue).trim();
    if (value.length < 8 || /\s/.test(value)) return message.error(t('resources.invalidToken'));
    try {
      if (scope === 'global') {
        await api.settings.huggingface.downloadToken.set(value);
        setGlobalValue('');
        await globalToken.refetch();
      } else {
        await api.settings.huggingface.publishToken.set(value);
        setPublishValue('');
        await publishToken.refetch();
      }
      message.success(t('resources.tokenSaved'));
    } catch (error) { message.error(error instanceof Error ? error.message : String(error)); }
  };

  const openAction = (next: ResourceAction) => {
    setAction(next);
    form.setFieldsValue({
      target_root: next.resource.install_root ?? next.resource.target_path,
      path: next.resource.target_path,
      preprocess_kind: next.resource.preprocess?.[0],
      gpu_count: 1,
    });
  };

  return (
    <div className="page resources-page">
      <PageIntro title={t('resources.title')} subtitle={t('resources.subtitle')} actions={<Button icon={<ReloadOutlined />} onClick={() => void refresh()}>{t('common.refresh')}</Button>} />
      <AsyncState loading={resources.isLoading || me.isLoading} error={resources.error ?? me.error} onRetry={() => void resources.refetch()}>
        <div className="card-grid two-columns">
          {me.data?.role === 'administrator' ? (
            <Card title={<Space><KeyOutlined />{t('resources.downloadToken')}</Space>} extra={<Tag color={globalToken.data?.configured ? 'green' : 'default'}>{t(globalToken.data?.configured ? 'common.enabled' : 'common.disabled')}</Tag>}>
              <Typography.Paragraph type="secondary">{t('resources.downloadTokenHint')}</Typography.Paragraph>
              <Space.Compact block><Input.Password value={globalValue} onChange={(event) => setGlobalValue(event.target.value)} /><Button type="primary" onClick={() => void saveToken('global')}>{t('common.save')}</Button></Space.Compact>
            </Card>
          ) : null}
          <Card title={<Space><KeyOutlined />{t('resources.publishToken')}</Space>} extra={<Tag color={publishToken.data?.configured ? 'green' : 'default'}>{t(publishToken.data?.configured ? 'common.enabled' : 'common.disabled')}</Tag>}>
            <Typography.Paragraph type="secondary">{t('resources.publishTokenHint')}</Typography.Paragraph>
            <Space.Compact block><Input.Password value={publishValue} onChange={(event) => setPublishValue(event.target.value)} /><Button type="primary" onClick={() => void saveToken('publish')}>{t('common.save')}</Button></Space.Compact>
          </Card>
        </div>
        <Card title={t('resources.inventory')}>
          <Table<ResourceRecord>
            rowKey="id"
            dataSource={resources.data?.items ?? []}
            pagination={false}
            columns={[
              { title: t('resources.resource'), dataIndex: 'name', render: (value, row) => <Space direction="vertical" size={0}><Typography.Text strong>{value}</Typography.Text><Typography.Text type="secondary">{row.kind}</Typography.Text></Space> },
              { title: t('common.status'), dataIndex: 'status', render: (value, row) => row.active_run ? <UtilityProgress run={row.active_run} compact /> : <Tag color={value === 'installed' ? 'green' : value === 'missing' ? 'orange' : 'default'}>{t(`resources.status.${value}`, { defaultValue: value })}</Tag> },
              { title: t('resources.path'), dataIndex: 'target_path', render: (value) => value || '—' },
              { title: t('common.actions'), render: (_, row) => <Space wrap>
                {row.installable && me.data?.role === 'administrator' ? <Button size="small" icon={<CloudDownloadOutlined />} title={row.requires_hf_token && !globalToken.data?.configured ? t('resources.tokenRequired') : undefined} disabled={Boolean(row.active_run) || Boolean(row.requires_hf_token && !globalToken.data?.configured)} onClick={() => openAction({ kind: 'install', resource: row })}>{t('resources.install')}</Button> : null}
                {row.active_run && ['queued', 'starting', 'running', 'stopping'].includes(row.active_run.status) ? <Button danger size="small" loading={cancelRun.isPending && cancelRun.variables === row.active_run.id} onClick={() => cancelRun.mutate(row.active_run!.id)}>{t('resources.cancelDownload')}</Button> : null}
                {row.registerable && me.data?.role === 'administrator' ? <Button size="small" icon={<FolderOpenOutlined />} onClick={() => openAction({ kind: 'register', resource: row })}>{t('resources.registerPath')}</Button> : null}
                {row.preprocess?.length ? <Button size="small" icon={<PlayCircleOutlined />} onClick={() => openAction({ kind: 'preprocess', resource: row })}>{t('resources.preprocess')}</Button> : null}
              </Space> },
            ]}
          />
        </Card>
        <Card title={t('resources.tasks')}>
          <Table rowKey="id" size="small" dataSource={utilities.data ?? []} pagination={{ pageSize: 8 }} columns={[
            { title: t('resources.taskType'), dataIndex: 'kind' },
            { title: t('common.status'), dataIndex: 'status', render: (value) => <Tag>{value}</Tag> },
            { title: t('resources.progress'), render: (_, row) => <UtilityProgress run={row} compact /> },
            { title: t('resources.queue'), render: (_, row) => `${row.queue_class.toUpperCase()}${row.queue_position ? ` · #${row.queue_position}` : ''}` },
            { title: t('resources.output'), dataIndex: 'output_path' },
            { title: t('common.actions'), render: (_, row) => <Space><Button size="small" href={api.utilities.logUrl(row.id)} target="_blank">Log</Button>{['queued', 'starting', 'running', 'stopping'].includes(row.status) ? <Button danger size="small" loading={cancelRun.isPending && cancelRun.variables === row.id} onClick={() => cancelRun.mutate(row.id)}>{t('common.cancel')}</Button> : null}</Space> },
          ]} />
        </Card>
      </AsyncState>
      <Modal open={Boolean(action)} title={action ? t(`resources.action.${action.kind}`) : ''} okText={t('common.confirm')} onCancel={() => { setAction(undefined); form.resetFields(); }} confirmLoading={submitAction.isPending} onOk={() => form.submit()}>
        {action?.kind === 'preprocess' ? <Alert showIcon type="warning" message={t('resources.gpuQueueHint')} style={{ marginBottom: 16 }} /> : null}
        <Form form={form} layout="vertical" onFinish={(values) => submitAction.mutate(values)}>
          {action?.kind === 'install' ? <Form.Item name="target_root" label={t('resources.targetRoot')} rules={[{ required: true }]}><Input /></Form.Item> : null}
          {action?.kind === 'register' ? <Form.Item name="path" label={t('resources.path')} rules={[{ required: true }]}><Input /></Form.Item> : null}
          {action?.kind === 'preprocess' ? <>
            <Form.Item name="preprocess_kind" label={t('resources.taskType')} rules={[{ required: true }]}><Select options={action.resource.preprocess?.map((value) => ({ value, label: value }))} /></Form.Item>
            <Form.Item name="data_root" label={t('resources.dataRoot')}><Input /></Form.Item>
            <Form.Item name="source_path" label={t('resources.sourcePath')}><Input /></Form.Item>
            <Form.Item name="output_path" label={t('resources.output')} rules={[{ required: true }]}><Input /></Form.Item>
            <Form.Item name="gpu_count" label="GPU" rules={[{ required: true }]}><InputNumber min={1} max={64} /></Form.Item>
          </> : null}
        </Form>
      </Modal>
    </div>
  );
}
