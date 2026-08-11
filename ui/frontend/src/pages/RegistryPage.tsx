import { CodeOutlined, DeleteOutlined, ExperimentOutlined, ReloadOutlined, SaveOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Descriptions, Input, Modal, Space, Statistic, Tabs, Tag, Typography, message } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { parse, stringify } from 'yaml';
import { api, isRecord } from '../api/client';
import type { Language, RegistryCombinationView, RegistryComponentView, RegistryDeploymentCombinationView, RegistryOverlayPreview } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';

function emptyOverlay(): Record<string, unknown> {
  const components = { backbones: [], action_heads: [], training_methods: [], datasets: [] };
  return {
    schema: 'alphabrain.registry-overlay',
    schema_version: 1,
    overlay_version: 'draft',
    additions: { components, combinations: [], deployment_combinations: [] },
    disables: { components: { ...components }, combinations: [], deployment_combinations: [] },
  };
}

function parseOverlay(value: string): Record<string, unknown> {
  const parsed: unknown = parse(value);
  if (!isRecord(parsed)) throw new Error('overlay_mapping_required');
  return parsed;
}

function statusColor(status: string): string {
  if (status === 'verified') return 'green';
  if (status === 'experimental') return 'gold';
  return 'default';
}

export function RegistryPage() {
  const { t, i18n } = useTranslation();
  const queryClient = useQueryClient();
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const registry = useQuery({ queryKey: ['registry'], queryFn: api.registry.browse });
  const isAdmin = me.data?.role === 'administrator';
  const overlay = useQuery({
    queryKey: ['registry-overlay'],
    queryFn: api.registry.overlay.get,
    enabled: isAdmin,
  });
  const [editor, setEditor] = useState(() => stringify(emptyOverlay()));
  const [preview, setPreview] = useState<RegistryOverlayPreview>();
  const language = (i18n.language === 'en-US' ? 'en-US' : 'zh-CN') as Language;

  useEffect(() => {
    if (overlay.data) setEditor(stringify(overlay.data.overlay ?? emptyOverlay()));
  }, [overlay.data]);

  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['registry'] }),
      queryClient.invalidateQueries({ queryKey: ['registry-overlay'] }),
    ]);
  };
  const previewMutation = useMutation({
    mutationFn: () => api.registry.overlay.preview(parseOverlay(editor)),
    onSuccess: (value) => { setPreview(value); message.success(t('registry.previewReady')); },
    onError: (error) => message.error(error instanceof Error ? error.message : String(error)),
  });
  const saveMutation = useMutation({
    mutationFn: () => api.registry.overlay.save(parseOverlay(editor)),
    onSuccess: async () => { setPreview(undefined); message.success(t('registry.saved')); await refresh(); },
    onError: (error) => message.error(error instanceof Error ? error.message : String(error)),
  });
  const deleteMutation = useMutation({
    mutationFn: api.registry.overlay.remove,
    onSuccess: async () => { setPreview(undefined); message.success(t('registry.deleted')); await refresh(); },
    onError: (error) => message.error(error instanceof Error ? error.message : String(error)),
  });
  const confirmDelete = () => Modal.confirm({
    title: t('registry.deleteTitle'),
    content: t('registry.deleteDescription'),
    okText: t('common.delete'),
    okButtonProps: { danger: true },
    onOk: () => deleteMutation.mutateAsync(),
  });

  const view = registry.data?.view;
  const componentGroups = useMemo(() => {
    const rows = view?.components ?? [];
    return ['backbones', 'action_heads', 'training_methods', 'datasets'].map((category) => ({
      category,
      rows: rows.filter((row) => row.category === category),
    }));
  }, [view?.components]);
  const localText = (value: Partial<Record<Language, string>> | undefined, fallback: string) =>
    value?.[language] || value?.['en-US'] || value?.['zh-CN'] || fallback;

  return (
    <div className="page registry-page">
      <PageIntro
        title={t('registry.title')}
        subtitle={t('registry.subtitle')}
        actions={<Button icon={<ReloadOutlined />} onClick={() => void refresh()}>{t('common.refresh')}</Button>}
      />
      <AsyncState loading={registry.isLoading || me.isLoading} error={registry.error ?? me.error} onRetry={() => void registry.refetch()}>
        {view ? (
          <>
            <div className="card-grid four-columns">
              <Card><Statistic title={t('registry.components')} value={view.summary.component_count} /></Card>
              <Card><Statistic title={t('registry.trainingCombinations')} value={view.summary.combination_count} /></Card>
              <Card><Statistic title={t('registry.deploymentCombinations')} value={view.summary.deployment_combination_count} /></Card>
              <Card><Statistic title={t('registry.overlayEntries')} value={view.summary.overlay_component_count + view.summary.overlay_combination_count + view.summary.overlay_deployment_combination_count} /></Card>
            </div>
            {view.warnings.length ? <Alert showIcon type="warning" message={t('registry.referenceWarning')} description={view.warnings.map((item) => `${item.code}: ${item.id ?? ''}`).join('\n')} /> : null}
            <Card title={<Space><ExperimentOutlined />{t('registry.inventory')}</Space>}>
              <Tabs items={[
                ...componentGroups.map(({ category, rows }) => ({
                  key: category,
                  label: t(`registry.category.${category}`),
                  children: <Table<RegistryComponentView>
                    rowKey="id"
                    size="small"
                    dataSource={rows}
                    pagination={{ pageSize: 12 }}
                    columns={[
                      { title: t('registry.entry'), render: (_, row) => <Space direction="vertical" size={0}><Typography.Text strong>{localText(row.label, row.id)}</Typography.Text><Typography.Text code>{row.id}</Typography.Text></Space> },
                      { title: t('common.status'), dataIndex: 'status', render: (value: string) => <Tag color={statusColor(value)}>{t(`capability.${value}`, { defaultValue: value })}</Tag> },
                      { title: t('registry.origin'), dataIndex: 'origin', render: (value: string) => <Tag>{t(`registry.${value}`)}</Tag> },
                      { title: t('registry.description'), render: (_, row) => localText(row.description, '—') },
                    ]}
                  />,
                })),
                {
                  key: 'training-combinations',
                  label: t('registry.trainingCombinations'),
                  children: <Table<RegistryCombinationView>
                    rowKey="id"
                    size="small"
                    scroll={{ x: 900 }}
                    dataSource={view.combinations}
                    pagination={{ pageSize: 12 }}
                    columns={[
                      { title: 'ID', dataIndex: 'id', render: (value: string) => <Typography.Text code>{value}</Typography.Text> },
                      { title: t('registry.model'), render: (_, row) => `${row.backbone} + ${row.action_head}` },
                      { title: t('registry.method'), dataIndex: 'method' },
                      { title: t('registry.datasets'), dataIndex: 'datasets', render: (values: string[]) => <Space wrap>{values.map((value) => <Tag key={value}>{value}</Tag>)}</Space> },
                      { title: 'GPU', dataIndex: 'min_gpus' },
                      { title: t('common.status'), dataIndex: 'status', render: (value: string) => <Tag color={statusColor(value)}>{t(`capability.${value}`, { defaultValue: value })}</Tag> },
                      { title: t('registry.origin'), dataIndex: 'origin', render: (value: string) => <Tag>{t(`registry.${value}`)}</Tag> },
                    ]}
                  />,
                },
                {
                  key: 'deployment-combinations',
                  label: t('registry.deploymentCombinations'),
                  children: <Table<RegistryDeploymentCombinationView>
                    rowKey="id"
                    size="small"
                    scroll={{ x: 900 }}
                    dataSource={view.deployment_combinations}
                    pagination={{ pageSize: 12 }}
                    columns={[
                      { title: 'ID', dataIndex: 'id', render: (value: string) => <Typography.Text code>{value}</Typography.Text> },
                      { title: t('registry.model'), render: (_, row) => `${row.backbone} + ${row.action_head}` },
                      { title: t('registry.adapter'), dataIndex: 'adapter' },
                      { title: t('registry.benchmarks'), dataIndex: 'benchmarks', render: (values: string[]) => <Space wrap>{values.map((value) => <Tag key={value}>{value}</Tag>)}</Space> },
                      { title: 'GPU', dataIndex: 'recommended_gpu_count' },
                      { title: t('common.status'), dataIndex: 'status', render: (value: string) => <Tag color={statusColor(value)}>{t(`capability.${value}`, { defaultValue: value })}</Tag> },
                    ]}
                  />,
                },
              ]} />
            </Card>
          </>
        ) : null}
      </AsyncState>
      {isAdmin ? (
          <Card title={<Space><CodeOutlined />{t('registry.overlayTitle')}</Space>} extra={<Tag color={overlay.data?.status.configured ? 'gold' : 'default'}>{t(overlay.data?.status.configured ? 'common.enabled' : 'common.disabled')}</Tag>}>
            <Alert showIcon type="warning" message={t('registry.overlayRiskTitle')} description={t('registry.overlayRiskDescription')} style={{ marginBottom: 16 }} />
            {overlay.error || registry.error ? <Alert showIcon type="error" message={t('registry.overlayUnreadable')} description={t('registry.overlayRecovery')} style={{ marginBottom: 16 }} /> : null}
            <Descriptions size="small" column={{ xs: 1, md: 3 }} style={{ marginBottom: 16 }}>
              <Descriptions.Item label={t('registry.version')}>{overlay.data?.status.overlay_version ?? '—'}</Descriptions.Item>
              <Descriptions.Item label="SHA-256">{overlay.data?.status.sha256?.slice(0, 16) ?? '—'}</Descriptions.Item>
              <Descriptions.Item label={t('registry.disabledEntries')}>{overlay.data?.status.disabled_entries ?? 0}</Descriptions.Item>
            </Descriptions>
            <Input.TextArea value={editor} onChange={(event) => { setEditor(event.target.value); setPreview(undefined); }} autoSize={{ minRows: 18, maxRows: 36 }} spellCheck={false} className="config-preview" />
            <Space wrap style={{ marginTop: 16 }}>
              <Button icon={<ExperimentOutlined />} loading={previewMutation.isPending} onClick={() => previewMutation.mutate()}>{t('registry.preview')}</Button>
              <Button type="primary" icon={<SaveOutlined />} loading={saveMutation.isPending} onClick={() => saveMutation.mutate()}>{t('common.save')}</Button>
              <Button danger icon={<DeleteOutlined />} disabled={!overlay.data?.status.configured && !overlay.error && !registry.error} loading={deleteMutation.isPending} onClick={confirmDelete}>{t('common.delete')}</Button>
            </Space>
            {preview ? <Alert showIcon type="success" style={{ marginTop: 16 }} message={t('registry.previewSummary', { components: preview.view.summary.overlay_component_count, combinations: preview.view.summary.overlay_combination_count, deployments: preview.view.summary.overlay_deployment_combination_count })} /> : null}
          </Card>
      ) : null}
    </div>
  );
}
