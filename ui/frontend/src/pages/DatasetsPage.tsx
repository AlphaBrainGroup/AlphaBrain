import { DeleteOutlined, EyeOutlined, PlusOutlined, ReloadOutlined, SearchOutlined, SettingOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Descriptions, Form, Input, InputNumber, Modal, Popconfirm, Radio, Select, Space, Tabs, Tag, Tooltip, Typography, message } from 'antd';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import type { DatasetBuilderSupport, DatasetInspectionResult, DatasetMixture, DatasetMixtureMember, DatasetRegistration } from '../api/types';
import { usePreferences } from '../app-context';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';

interface RegisterFormValues {
  name: string;
  path: string;
  description?: string;
  storage_mode: 'reference' | 'managed_copy';
  visibility: 'private' | 'shared';
}

interface MixtureFormValues {
  name: string;
  visibility: 'private' | 'shared';
  members: DatasetMixtureMember[];
}

const PAGE_SIZES = [20, 50, 100];
const FORMAT_FAMILIES = ['lerobot', 'lerobot_collection', 'cosmos', 'vlm_json'];

function bytes(value: number) {
  if (!value) return '0 B';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function queryChoice(value: string | null, choices: readonly string[]): string {
  return value && choices.includes(value) ? value : 'all';
}

function formatFamily(row: DatasetRegistration): string {
  const value = row.validation?.format_family;
  return typeof value === 'string' ? value : 'unknown';
}

function builderSupport(row: DatasetRegistration): DatasetBuilderSupport {
  const value = row.validation?.builder_support;
  return value === 'direct' || value === 'mixture_only' || value === 'inventory_only' ? value : 'unsupported';
}

function mixtureHealth(row: DatasetMixture): 'ready' | 'degraded' | 'missing' {
  if (!row.resolved_members.length || row.resolved_members.some((item) => !item.dataset_path || item.dataset_status === 'missing')) return 'missing';
  return row.resolved_members.every((item) => item.dataset_status === 'ready') ? 'ready' : 'degraded';
}

function memberBucket(count: number): 'one' | 'few' | 'many' {
  if (count <= 1) return 'one';
  return count <= 5 ? 'few' : 'many';
}

function pageValue(value: string | null): number {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : 1;
}

function pageSizeValue(value: string | null): number {
  const parsed = Number(value);
  return PAGE_SIZES.includes(parsed) ? parsed : PAGE_SIZES[0];
}

export function DatasetsPage() {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const datasets = useQuery({ queryKey: ['registered-datasets'], queryFn: api.datasets.list, refetchInterval: 5_000 });
  const mixtures = useQuery({ queryKey: ['dataset-mixtures'], queryFn: api.datasets.mixtures.list });
  const [registerOpen, setRegisterOpen] = useState(false);
  const [mixtureOpen, setMixtureOpen] = useState(false);
  const [previewId, setPreviewId] = useState<string>();
  const [inspection, setInspection] = useState<DatasetInspectionResult>();
  const [inspectedPath, setInspectedPath] = useState('');
  const preview = useQuery({ queryKey: ['dataset-preview', previewId], queryFn: () => api.datasets.preview(previewId!), enabled: Boolean(previewId) });
  const [registerForm] = Form.useForm<RegisterFormValues>();
  const [mixtureForm] = Form.useForm<MixtureFormValues>();
  const registerPath = Form.useWatch('path', registerForm) ?? '';
  const mixtureMembers = Form.useWatch('members', mixtureForm) ?? [];

  const refresh = async () => Promise.all([
    queryClient.invalidateQueries({ queryKey: ['registered-datasets'] }),
    queryClient.invalidateQueries({ queryKey: ['dataset-mixtures'] }),
  ]);
  const register = useMutation({
    mutationFn: api.datasets.register,
    onSuccess: async () => {
      message.success(t('datasets.registered'));
      setRegisterOpen(false);
      setInspection(undefined);
      setInspectedPath('');
      registerForm.resetFields();
      await refresh();
    },
    onError: (error) => message.error(error.message),
  });
  const inspectMutation = useMutation({
    mutationFn: api.datasets.inspect,
    onSuccess: (result, path) => {
      setInspection(result);
      setInspectedPath(path);
    },
    onError: (error) => {
      setInspection(undefined);
      setInspectedPath('');
      message.error(error instanceof Error ? error.message : String(error));
    },
  });
  const createMixture = useMutation({ mutationFn: api.datasets.mixtures.create, onSuccess: async () => { message.success(t('datasets.mixtureCreated')); setMixtureOpen(false); mixtureForm.resetFields(); await refresh(); }, onError: (error) => message.error(error.message) });
  const remove = async (row: DatasetRegistration) => {
    try { await api.datasets.remove(row.id); message.success(t('datasets.unregistered')); await refresh(); } catch (error) { message.error(error instanceof Error ? error.message : String(error)); }
  };
  const removeMixture = async (row: DatasetMixture) => {
    try { await api.datasets.mixtures.remove(row.id); await refresh(); } catch (error) { message.error(error instanceof Error ? error.message : String(error)); }
  };

  const allDatasets = datasets.data ?? [];
  const allMixtures = mixtures.data ?? [];
  const availableDatasets = allDatasets.filter((row) => row.status === 'ready');
  const activeTab = searchParams.get('tab') === 'mixtures' ? 'mixtures' : 'datasets';

  const dsQuery = searchParams.get('ds_q') ?? '';
  const dsFormat = queryChoice(searchParams.get('ds_format'), FORMAT_FAMILIES);
  const dsStatus = queryChoice(searchParams.get('ds_status'), ['ready', 'copying', 'invalid']);
  const dsSupport = queryChoice(searchParams.get('ds_support'), ['direct', 'mixture_only', 'inventory_only']);
  const dsVisibility = queryChoice(searchParams.get('ds_visibility'), ['private', 'shared']);
  const dsSource = queryChoice(searchParams.get('ds_source'), ['mine', 'others']);
  const dsStorage = queryChoice(searchParams.get('ds_storage'), ['reference', 'managed_copy']);
  const dsStats = queryChoice(searchParams.get('ds_stats'), ['unknown', 'queued', 'ready', 'stale', 'failed']);
  const dsPage = pageValue(searchParams.get('ds_page'));
  const dsPageSize = pageSizeValue(searchParams.get('ds_page_size'));

  const filteredDatasets = useMemo(() => {
    const term = dsQuery.trim().toLocaleLowerCase();
    return allDatasets.filter((row) => {
      const sourceMatches = dsSource === 'all'
        || (dsSource === 'mine' && row.owner_id === me.data?.id)
        || (dsSource === 'others' && row.owner_id !== me.data?.id && row.visibility === 'shared');
      const text = [row.name, row.description, row.path, row.source_path, row.owner_name, row.format, formatFamily(row)].filter(Boolean).join(' ').toLocaleLowerCase();
      return sourceMatches
        && (dsFormat === 'all' || formatFamily(row) === dsFormat)
        && (dsStatus === 'all' || row.status === dsStatus)
        && (dsSupport === 'all' || builderSupport(row) === dsSupport)
        && (dsVisibility === 'all' || row.visibility === dsVisibility)
        && (dsStorage === 'all' || row.storage_mode === dsStorage)
        && (dsStats === 'all' || (row.stats_status || 'unknown') === dsStats)
        && (!term || text.includes(term));
    });
  }, [allDatasets, dsFormat, dsQuery, dsSource, dsStats, dsStatus, dsStorage, dsSupport, dsVisibility, me.data?.id]);

  const mixQuery = searchParams.get('mix_q') ?? '';
  const mixVisibility = queryChoice(searchParams.get('mix_visibility'), ['private', 'shared']);
  const mixSource = queryChoice(searchParams.get('mix_source'), ['mine', 'others']);
  const mixHealth = queryChoice(searchParams.get('mix_health'), ['ready', 'degraded', 'missing']);
  const mixFormat = queryChoice(searchParams.get('mix_format'), [...FORMAT_FAMILIES, 'mixed']);
  const mixMembers = queryChoice(searchParams.get('mix_members'), ['one', 'few', 'many']);
  const mixPage = pageValue(searchParams.get('mix_page'));
  const mixPageSize = pageSizeValue(searchParams.get('mix_page_size'));

  const filteredMixtures = useMemo(() => {
    const term = mixQuery.trim().toLocaleLowerCase();
    return allMixtures.filter((row) => {
      const families = new Set(row.resolved_members.map((item) => item.format_family ?? 'unknown'));
      const sourceMatches = mixSource === 'all'
        || (mixSource === 'mine' && row.owner_id === me.data?.id)
        || (mixSource === 'others' && row.owner_id !== me.data?.id && row.visibility === 'shared');
      const formatMatches = mixFormat === 'all'
        || (mixFormat === 'mixed' ? families.size > 1 : families.has(mixFormat));
      const text = [
        row.name, row.description, row.owner_name,
        ...row.resolved_members.flatMap((item) => [item.dataset_name, item.dataset_format, item.format_family, item.robot_type, item.pattern]),
      ].filter(Boolean).join(' ').toLocaleLowerCase();
      return sourceMatches
        && (mixVisibility === 'all' || row.visibility === mixVisibility)
        && (mixHealth === 'all' || mixtureHealth(row) === mixHealth)
        && formatMatches
        && (mixMembers === 'all' || memberBucket(row.resolved_members.length) === mixMembers)
        && (!term || text.includes(term));
    });
  }, [allMixtures, me.data?.id, mixFormat, mixHealth, mixMembers, mixQuery, mixSource, mixVisibility]);

  const setFilter = (key: string, value: string | undefined, pageKey: string) => {
    const next = new URLSearchParams(searchParams);
    if (!value || value === 'all') next.delete(key);
    else next.set(key, value);
    next.delete(pageKey);
    setSearchParams(next, { replace: true });
  };
  const clearFilters = (prefix: 'ds_' | 'mix_') => {
    const next = new URLSearchParams(searchParams);
    Array.from(next.keys()).filter((key) => key.startsWith(prefix)).forEach((key) => next.delete(key));
    setSearchParams(next, { replace: true });
  };
  const setTab = (tab: string) => {
    const next = new URLSearchParams(searchParams);
    if (tab === 'datasets') next.delete('tab');
    else next.set('tab', tab);
    setSearchParams(next, { replace: true });
  };
  const setPagination = (prefix: 'ds' | 'mix', nextPage: number, nextSize: number, currentSize: number) => {
    const next = new URLSearchParams(searchParams);
    if (nextSize === PAGE_SIZES[0]) next.delete(`${prefix}_page_size`);
    else next.set(`${prefix}_page_size`, String(nextSize));
    if (nextPage === 1 || nextSize !== currentSize) next.delete(`${prefix}_page`);
    else next.set(`${prefix}_page`, String(nextPage));
    setSearchParams(next, { replace: true });
  };

  const supportTag = (support: DatasetBuilderSupport) => <Tag color={support === 'direct' ? 'green' : support === 'mixture_only' ? 'gold' : support === 'inventory_only' ? 'blue' : 'red'}>{t(`datasets.builderSupport.${support}`)}</Tag>;
  const familyTag = (family: string, label?: string) => <Tag color={family.startsWith('lerobot') ? 'cyan' : family === 'cosmos' ? 'purple' : family === 'vlm_json' ? 'blue' : 'default'}>{label ?? t(`datasets.formatFamily.${family}`)}</Tag>;
  const statusTag = (status: string) => <Tag color={status === 'ready' ? 'green' : status === 'invalid' || status === 'failed' ? 'red' : 'processing'}>{t(`datasets.status.${status}`, { defaultValue: status })}</Tag>;

  const datasetTable = <Table<DatasetRegistration>
    rowKey="id"
    dataSource={filteredDatasets}
    locale={{ emptyText: allDatasets.length ? t('datasets.noFilterResults') : t('common.noData') }}
    scroll={{ x: 1250 }}
    pagination={{
      current: Math.min(dsPage, Math.max(1, Math.ceil(filteredDatasets.length / dsPageSize))),
      pageSize: dsPageSize,
      total: filteredDatasets.length,
      showSizeChanger: true,
      pageSizeOptions: PAGE_SIZES.map(String),
      showTotal: (total) => t('datasets.paginationTotal', { total }),
      onChange: (page, size) => setPagination('ds', page, size, dsPageSize),
    }}
    columns={[
      { title: t('datasets.name'), dataIndex: 'name', fixed: 'left', render: (value, row) => <Space direction="vertical" size={0}><Typography.Text strong>{value}</Typography.Text><Typography.Text type="secondary" ellipsis>{row.path}</Typography.Text></Space> },
      { title: t('datasets.format'), dataIndex: 'format', render: (value, row) => <Space size={[4, 4]} wrap>{familyTag(formatFamily(row), value)}{supportTag(builderSupport(row))}</Space> },
      { title: t('common.status'), dataIndex: 'status', render: statusTag },
      { title: t('datasets.episodes'), dataIndex: 'episode_count', responsive: ['lg'] },
      { title: t('datasets.size'), dataIndex: 'size_bytes', render: bytes, responsive: ['lg'] },
      { title: t('datasets.storageMode'), dataIndex: 'storage_mode', render: (value) => t(`datasets.${value}`), responsive: ['xl'] },
      { title: t('datasets.sharing'), dataIndex: 'visibility', render: (value) => <Tag>{t(`datasets.${value}`)}</Tag>, responsive: ['md'] },
      { title: t('common.owner'), dataIndex: 'owner_name', responsive: ['xl'] },
      { title: t('common.actions'), fixed: 'right', render: (_, row) => <Space wrap>
        <Button size="small" icon={<EyeOutlined />} disabled={row.status !== 'ready'} onClick={() => setPreviewId(row.id)}>{t('common.view')}</Button>
        <Button size="small" icon={<ReloadOutlined />} onClick={() => void api.datasets.revalidate(row.id).then(refresh)}>{t('datasets.validate')}</Button>
        <Button size="small" icon={<SettingOutlined />} disabled={row.status !== 'ready'} onClick={() => void api.datasets.stats(row.id).then(refresh)}>{t('datasets.stats')}</Button>
        <Popconfirm title={t('datasets.removeConfirm')} onConfirm={() => void remove(row)}><Button danger size="small" icon={<DeleteOutlined />}>{t('common.delete')}</Button></Popconfirm>
      </Space> },
    ]}
  />;

  const mixtureTable = <Table<DatasetMixture>
    rowKey="id"
    dataSource={filteredMixtures}
    locale={{ emptyText: allMixtures.length ? t('datasets.noMixtureFilterResults') : t('common.noData') }}
    scroll={{ x: 950 }}
    pagination={{
      current: Math.min(mixPage, Math.max(1, Math.ceil(filteredMixtures.length / mixPageSize))),
      pageSize: mixPageSize,
      total: filteredMixtures.length,
      showSizeChanger: true,
      pageSizeOptions: PAGE_SIZES.map(String),
      showTotal: (total) => t('datasets.paginationTotal', { total }),
      onChange: (page, size) => setPagination('mix', page, size, mixPageSize),
    }}
    columns={[
      { title: t('datasets.name'), dataIndex: 'name', fixed: 'left', render: (value, row) => <Space direction="vertical" size={0}><Typography.Text strong>{value}</Typography.Text><Typography.Text type="secondary">{row.owner_name}</Typography.Text></Space> },
      { title: t('datasets.members'), render: (_, row) => <Space size={[4, 4]} wrap>{row.resolved_members.map((item) => <Tag key={item.registration_id}>{item.dataset_name ?? item.registration_id}</Tag>)}</Space> },
      { title: t('datasets.memberFormats'), render: (_, row) => <Space size={[4, 4]} wrap>{Array.from(new Set(row.resolved_members.map((item) => item.format_family ?? 'unknown'))).map((family) => familyTag(family))}</Space> },
      { title: t('datasets.mixtureHealth'), render: (_, row) => statusTag(mixtureHealth(row)) },
      { title: t('datasets.sharing'), dataIndex: 'visibility', render: (value) => <Tag>{t(`datasets.${value}`)}</Tag> },
      { title: t('datasets.version'), dataIndex: 'version', render: (value) => `v${value}` },
      { title: t('common.actions'), fixed: 'right', render: (_, row) => <Popconfirm title={t('datasets.removeMixtureConfirm')} onConfirm={() => void removeMixture(row)}><Button danger size="small" icon={<DeleteOutlined />}>{t('common.delete')}</Button></Popconfirm> },
    ]}
  />;

  const datasetFilters = <div className="dataset-filter-toolbar">
    <Input className="table-search" allowClear prefix={<SearchOutlined />} placeholder={t('common.search')} value={dsQuery} onChange={(event) => setFilter('ds_q', event.target.value, 'ds_page')} />
    <div className="dataset-filter-controls">
      <Select value={dsFormat} onChange={(value) => setFilter('ds_format', value, 'ds_page')} options={[{ value: 'all', label: t('datasets.allFormats') }, ...FORMAT_FAMILIES.map((value) => ({ value, label: t(`datasets.formatFamily.${value}`) }))]} />
      <Select value={dsStatus} onChange={(value) => setFilter('ds_status', value, 'ds_page')} options={[{ value: 'all', label: t('datasets.allStatuses') }, ...['ready', 'copying', 'invalid'].map((value) => ({ value, label: t(`datasets.status.${value}`) }))]} />
      <Select value={dsSupport} onChange={(value) => setFilter('ds_support', value, 'ds_page')} options={[{ value: 'all', label: t('datasets.allBuilderSupport') }, ...['direct', 'mixture_only', 'inventory_only'].map((value) => ({ value, label: t(`datasets.builderSupport.${value}`) }))]} />
      <Select value={dsVisibility} onChange={(value) => setFilter('ds_visibility', value, 'ds_page')} options={[{ value: 'all', label: t('datasets.allVisibilities') }, { value: 'private', label: t('datasets.private') }, { value: 'shared', label: t('datasets.shared') }]} />
      <Select value={dsSource} onChange={(value) => setFilter('ds_source', value, 'ds_page')} options={[{ value: 'all', label: t('datasets.allSources') }, { value: 'mine', label: t('datasets.mine') }, { value: 'others', label: t('datasets.sharedByOthers') }]} />
      <Select value={dsStorage} onChange={(value) => setFilter('ds_storage', value, 'ds_page')} options={[{ value: 'all', label: t('datasets.allStorageModes') }, { value: 'reference', label: t('datasets.reference') }, { value: 'managed_copy', label: t('datasets.managedCopy') }]} />
      <Select value={dsStats} onChange={(value) => setFilter('ds_stats', value, 'ds_page')} options={[{ value: 'all', label: t('datasets.allStatsStatuses') }, ...['unknown', 'queued', 'ready', 'stale', 'failed'].map((value) => ({ value, label: t(`datasets.statsStatus.${value}`) }))]} />
    </div>
    <div className="dataset-filter-summary"><Typography.Text type="secondary">{t('datasets.resultCount', { count: filteredDatasets.length, total: allDatasets.length })}</Typography.Text><Button type="link" size="small" onClick={() => clearFilters('ds_')}>{t('datasets.clearFilters')}</Button></div>
  </div>;

  const mixtureFilters = <div className="dataset-filter-toolbar">
    <Input className="table-search" allowClear prefix={<SearchOutlined />} placeholder={t('common.search')} value={mixQuery} onChange={(event) => setFilter('mix_q', event.target.value, 'mix_page')} />
    <div className="dataset-filter-controls">
      <Select value={mixHealth} onChange={(value) => setFilter('mix_health', value, 'mix_page')} options={[{ value: 'all', label: t('datasets.allHealthStatuses') }, ...['ready', 'degraded', 'missing'].map((value) => ({ value, label: t(`datasets.status.${value}`) }))]} />
      <Select value={mixFormat} onChange={(value) => setFilter('mix_format', value, 'mix_page')} options={[{ value: 'all', label: t('datasets.allFormats') }, ...FORMAT_FAMILIES.map((value) => ({ value, label: t(`datasets.formatFamily.${value}`) })), { value: 'mixed', label: t('datasets.mixedFormats') }]} />
      <Select value={mixMembers} onChange={(value) => setFilter('mix_members', value, 'mix_page')} options={[{ value: 'all', label: t('datasets.allMemberCounts') }, ...['one', 'few', 'many'].map((value) => ({ value, label: t(`datasets.memberCount.${value}`) }))]} />
      <Select value={mixVisibility} onChange={(value) => setFilter('mix_visibility', value, 'mix_page')} options={[{ value: 'all', label: t('datasets.allVisibilities') }, { value: 'private', label: t('datasets.private') }, { value: 'shared', label: t('datasets.shared') }]} />
      <Select value={mixSource} onChange={(value) => setFilter('mix_source', value, 'mix_page')} options={[{ value: 'all', label: t('datasets.allSources') }, { value: 'mine', label: t('datasets.mine') }, { value: 'others', label: t('datasets.sharedByOthers') }]} />
    </div>
    <div className="dataset-filter-summary"><Typography.Text type="secondary">{t('datasets.resultCount', { count: filteredMixtures.length, total: allMixtures.length })}</Typography.Text><Button type="link" size="small" onClick={() => clearFilters('mix_')}>{t('datasets.clearFilters')}</Button></div>
  </div>;

  const selectedRegistrations = mixtureMembers.flatMap((member) => {
    const row = availableDatasets.find((item) => item.id === member?.registration_id);
    return row ? [row] : [];
  });
  const selectedFamilies = new Set(selectedRegistrations.map(formatFamily));
  const mixtureCompatibilityWarning = selectedFamilies.size > 1 || selectedRegistrations.some((row) => builderSupport(row) !== 'direct');
  const currentPathReady = Boolean(inspection?.valid && inspectedPath === registerPath.trim());

  return (
    <div className="page datasets-page">
      <PageIntro title={t('datasets.title')} subtitle={t('datasets.subtitle')} />
      <Alert
        className="dataset-support-alert"
        type="info"
        showIcon
        message={t('datasets.supportedFormatsTitle')}
        description={<Space direction="vertical" size={6}><Space wrap>{familyTag('lerobot', 'LeRobot v2/v3')}{familyTag('lerobot_collection')}{familyTag('cosmos')}{familyTag('vlm_json')}</Space><Typography.Text type="secondary">{t('datasets.supportedFormatsHint')}</Typography.Text></Space>}
      />
      <AsyncState loading={datasets.isLoading || mixtures.isLoading || me.isLoading} error={datasets.error ?? mixtures.error ?? me.error} onRetry={() => void refresh()}>
        <Card>
          <Tabs activeKey={activeTab} onChange={setTab} items={[
            { key: 'datasets', label: t('datasets.registeredDatasets'), children: <><div className="dataset-tab-actions"><Button type="primary" icon={<PlusOutlined />} onClick={() => { registerForm.setFieldsValue({ storage_mode: 'reference', visibility: 'shared' }); setInspection(undefined); setInspectedPath(''); setRegisterOpen(true); }}>{t('datasets.register')}</Button></div>{datasetFilters}{datasetTable}</> },
            { key: 'mixtures', label: t('datasets.mixtures'), children: <><div className="dataset-tab-actions"><Tooltip title={!availableDatasets.length ? t('datasets.mixtureRequiresReady') : t('datasets.mixtureAvailableCount', { count: availableDatasets.length })}><span><Button type="primary" icon={<PlusOutlined />} disabled={!availableDatasets.length} onClick={() => { mixtureForm.setFieldsValue({ visibility: 'shared', members: [{ weight: 1 }] }); setMixtureOpen(true); }}>{t('datasets.createMixtureWithCount', { count: availableDatasets.length })}</Button></span></Tooltip></div>{mixtureFilters}{mixtureTable}</> },
          ]} />
        </Card>
      </AsyncState>
      <Modal
        width={760}
        open={registerOpen}
        title={t('datasets.register')}
        okText={t('common.confirm')}
        confirmLoading={register.isPending}
        okButtonProps={{ disabled: !currentPathReady }}
        onCancel={() => setRegisterOpen(false)}
        onOk={() => registerForm.submit()}
      >
        <Alert type="info" showIcon message={t('datasets.inspectBeforeRegister')} description={t('datasets.inspectBeforeRegisterHint')} />
        <Form<RegisterFormValues>
          form={registerForm}
          layout="vertical"
          onValuesChange={(changed) => {
            if ('path' in changed && String(changed.path ?? '').trim() !== inspectedPath) {
              setInspection(undefined);
              setInspectedPath('');
              inspectMutation.reset();
            }
          }}
          onFinish={(values) => {
            if (!currentPathReady) return;
            register.mutate(values);
          }}
        >
          <Form.Item name="name" label={t('datasets.name')} rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item label={t('datasets.path')} required extra={t('datasets.pathHint')}>
            <Space.Compact block>
              <Form.Item name="path" noStyle rules={[{ required: true }]}><Input /></Form.Item>
              <Button loading={inspectMutation.isPending} onClick={() => void registerForm.validateFields(['path']).then(({ path }) => inspectMutation.mutate(path.trim()))}>{t('datasets.detectFormat')}</Button>
            </Space.Compact>
          </Form.Item>
          {inspection ? <Alert
            className="dataset-inspection-result"
            type={inspection.valid ? inspection.builder_support === 'direct' ? 'success' : 'warning' : 'error'}
            showIcon
            message={inspection.valid ? t('datasets.detectedFormat', { format: inspection.format }) : t('datasets.unsupportedFormat')}
            description={<Space direction="vertical" size={6} className="full-width">
              <Space wrap>{familyTag(inspection.format_family, inspection.format)}{supportTag(inspection.builder_support)}{inspection.compatible_loaders.map((loader) => <Tag key={loader}>{loader}</Tag>)}</Space>
              <Typography.Text type="secondary">{t(`datasets.builderSupportHint.${inspection.builder_support}`)}</Typography.Text>
              <Typography.Text type="secondary">{t('datasets.inspectionCounts', { datasets: inspection.dataset_count, episodes: inspection.episode_count, parquet: inspection.parquet_count, size: bytes(inspection.size_bytes) })}</Typography.Text>
              {inspection.issues.map((issue) => <Typography.Text key={`${issue.id}-${issue.path}`} type={issue.level === 'error' ? 'danger' : 'warning'}>{issue.message_i18n?.[language] ?? issue.message ?? issue.title}</Typography.Text>)}
            </Space>}
          /> : null}
          <Form.Item name="description" label={t('datasets.description')}><Input.TextArea /></Form.Item>
          <Form.Item name="storage_mode" label={t('datasets.storageMode')}><Radio.Group><Radio value="reference">{t('datasets.reference')}</Radio><Radio value="managed_copy">{t('datasets.managedCopy')}</Radio></Radio.Group></Form.Item>
          <Form.Item name="visibility" label={t('datasets.sharing')}><Radio.Group><Radio value="shared">{t('datasets.shared')}</Radio><Radio value="private">{t('datasets.private')}</Radio></Radio.Group></Form.Item>
        </Form>
      </Modal>
      <Modal width={760} open={mixtureOpen} title={t('datasets.createMixture')} okText={t('common.create')} confirmLoading={createMixture.isPending} onCancel={() => setMixtureOpen(false)} onOk={() => mixtureForm.submit()}>
        <Form<MixtureFormValues> form={mixtureForm} layout="vertical" onFinish={(values) => createMixture.mutate(values)}>
          <Form.Item name="name" label={t('datasets.name')} rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="visibility" label={t('datasets.sharing')}><Radio.Group><Radio value="shared">{t('datasets.shared')}</Radio><Radio value="private">{t('datasets.private')}</Radio></Radio.Group></Form.Item>
          {mixtureCompatibilityWarning ? <Alert className="dataset-mixture-warning" type="warning" showIcon message={t('datasets.mixedFormatWarning')} description={t('datasets.mixedFormatWarningHint')} /> : null}
          <Form.List name="members">{(fields, { add, remove: removeField }) => <Space direction="vertical" className="full-width">
            {fields.map((field) => <Card size="small" key={field.key} extra={fields.length > 1 ? <Button type="text" danger onClick={() => removeField(field.name)}>{t('common.delete')}</Button> : null}>
              <Form.Item {...field} name={[field.name, 'registration_id']} label={t('datasets.dataset')} rules={[{ required: true }]}><Select options={availableDatasets.map((row) => ({ value: row.id, label: `${row.name} · ${row.format} · ${t(`datasets.builderSupport.${builderSupport(row)}`)}` }))} /></Form.Item>
              <div className="form-grid-2"><Form.Item name={[field.name, 'weight']} label={t('datasets.weight')} rules={[{ required: true }]}><InputNumber min={0.0001} className="full-width" /></Form.Item><Form.Item name={[field.name, 'robot_type']} label={t('datasets.robotType')}><Input /></Form.Item></div>
              <Form.Item name={[field.name, 'pattern']} label={t('datasets.pattern')} extra={t('datasets.patternHint')}><Input /></Form.Item>
            </Card>)}
            <Button block onClick={() => add({ weight: 1 })}>{t('datasets.addMember')}</Button>
          </Space>}</Form.List>
        </Form>
      </Modal>
      <Modal width={900} open={Boolean(previewId)} footer={null} title={t('datasets.preview')} onCancel={() => setPreviewId(undefined)}>
        <AsyncState loading={preview.isLoading} error={preview.error}><Descriptions bordered size="small" column={2} items={Object.entries(preview.data?.summary ?? {}).filter(([, value]) => ['string', 'number', 'boolean'].includes(typeof value)).slice(0, 12).map(([key, value]) => ({ key, label: key, children: String(value) }))} /><Typography.Title level={5}>{t('datasets.sampleMetadata')}</Typography.Title><pre className="code-block">{JSON.stringify({ tasks: preview.data?.tasks, episodes: preview.data?.episodes }, null, 2)}</pre></AsyncState>
      </Modal>
    </div>
  );
}
