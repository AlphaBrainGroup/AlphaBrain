import { CopyOutlined, DeleteOutlined, EditOutlined, PlusOutlined, RocketOutlined, SearchOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Button, Card, Empty, Form, Input, Modal, Radio, Select, Space, Tag, Typography, message } from 'antd';
import dayjs from 'dayjs';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { api, listFrom, templateDuplicateDetail } from '../api/client';
import type { ExperimentSpec, Template, TemplateDuplicateDetail, TemplateDuplicateMatch } from '../api/types';
import { usePreferences } from '../app-context';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { TemplateDuplicateModal } from '../components/TemplateDuplicateModal';

interface TemplateForm {
  name: string;
  description?: string;
  visibility: 'personal' | 'shared';
}

interface SaveTemplateRequest {
  values: TemplateForm;
  allowDuplicate: boolean;
  editingId?: string;
  spec?: ExperimentSpec;
}

interface PendingDuplicate {
  detail: TemplateDuplicateDetail;
  request: SaveTemplateRequest;
}

const PAGE_SIZES = [20, 50, 100];

function queryChoice(value: string | null, choices: readonly string[]): string {
  return value && choices.includes(value) ? value : 'all';
}

export function TemplatesPage() {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<Template>();
  const [cloning, setCloning] = useState<Template>();
  const [pendingDuplicate, setPendingDuplicate] = useState<PendingDuplicate>();
  const [form] = Form.useForm<TemplateForm>();
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const templates = useQuery({ queryKey: ['templates'], queryFn: api.templates.list });

  const save = useMutation({
    mutationFn: (request: SaveTemplateRequest) => request.editingId
      ? api.templates.update(request.editingId, request.values, request.allowDuplicate)
      : api.templates.create({ ...request.values, spec: request.spec }, request.allowDuplicate),
    onSuccess: async () => {
      setPendingDuplicate(undefined);
      setOpen(false);
      setEditing(undefined);
      setCloning(undefined);
      form.resetFields();
      await queryClient.invalidateQueries({ queryKey: ['templates'] });
    },
    onError: (error, request) => {
      const detail = templateDuplicateDetail(error);
      if (detail) setPendingDuplicate({ detail, request });
      else message.error(error instanceof Error ? error.message : String(error));
    },
  });
  const remove = useMutation({
    mutationFn: api.templates.remove,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['templates'] }),
    onError: (error) => message.error(error.message),
  });

  const allRows = listFrom(templates.data);
  const search = searchParams.get('q') ?? '';
  const source = queryChoice(searchParams.get('source'), ['builtin', 'mine', 'shared']);
  const visibility = queryChoice(searchParams.get('visibility'), ['personal', 'shared']);
  const architecture = searchParams.get('backbone') ?? 'all';
  const method = searchParams.get('method') ?? 'all';
  const dataset = searchParams.get('dataset') ?? 'all';
  const availability = queryChoice(searchParams.get('availability'), ['ready', 'partial', 'missing', 'custom']);
  const category = searchParams.get('category') ?? 'all';
  const requestedPageSize = Number(searchParams.get('page_size'));
  const pageSize = PAGE_SIZES.includes(requestedPageSize) ? requestedPageSize : PAGE_SIZES[0];
  const requestedPage = Number(searchParams.get('page'));
  const page = Number.isInteger(requestedPage) && requestedPage > 0 ? requestedPage : 1;

  const options = useMemo(() => {
    const values = (field: keyof Pick<Template, 'architecture' | 'method' | 'dataset' | 'category'>) =>
      Array.from(new Set(allRows.map((item) => item[field]).filter((value): value is string => Boolean(value)))).sort((a, b) => a.localeCompare(b));
    return {
      architectures: values('architecture'),
      methods: values('method'),
      datasets: values('dataset'),
      categories: values('category'),
    };
  }, [allRows]);

  const rows = useMemo(() => {
    const term = search.trim().toLocaleLowerCase();
    return allRows.filter((item) => {
      const sourceMatches = source === 'all'
        || (source === 'builtin' && item.builtin)
        || (source === 'mine' && !item.builtin && item.owner_id === me.data?.id)
        || (source === 'shared' && !item.builtin && item.visibility === 'shared' && item.owner_id !== me.data?.id);
      const itemAvailability = item.availability ?? 'custom';
      const text = [
        item.name,
        item.name_i18n?.['zh-CN'],
        item.name_i18n?.['en-US'],
        item.description,
        item.description_i18n?.['zh-CN'],
        item.description_i18n?.['en-US'],
        ...(item.tags ?? []),
        item.owner_name,
        item.architecture,
        item.method,
        item.dataset,
        item.category,
        item.visibility,
        itemAvailability,
        t(`templates.${item.visibility}`),
        item.availability ? t(`templates.availability.${item.availability}`) : t('templates.customTemplate'),
        item.builtin ? 'builtin' : '',
        item.builtin ? t('templates.builtin') : '',
      ].filter(Boolean).join(' ').toLocaleLowerCase();
      return sourceMatches
        && (visibility === 'all' || item.visibility === visibility)
        && (architecture === 'all' || item.architecture === architecture)
        && (method === 'all' || item.method === method)
        && (dataset === 'all' || item.dataset === dataset)
        && (availability === 'all' || itemAvailability === availability)
        && (category === 'all' || item.category === category)
        && (!term || text.includes(term));
    });
  }, [allRows, architecture, availability, category, dataset, me.data?.id, method, search, source, t, visibility]);

  const maxPage = Math.max(1, Math.ceil(rows.length / pageSize));
  useEffect(() => {
    if (page <= maxPage) return;
    const next = new URLSearchParams(searchParams);
    if (maxPage === 1) next.delete('page');
    else next.set('page', String(maxPage));
    setSearchParams(next, { replace: true });
  }, [maxPage, page, searchParams, setSearchParams]);

  const setFilter = (key: string, value?: string) => {
    const next = new URLSearchParams(searchParams);
    if (!value || value === 'all') next.delete(key);
    else next.set(key, value);
    next.delete('page');
    setSearchParams(next, { replace: true });
  };
  const clearFilters = () => setSearchParams({}, { replace: true });
  const hasFilters = Boolean(search.trim()) || [source, visibility, architecture, method, dataset, availability, category].some((value) => value !== 'all');
  const localizedName = (item: Template) => item.name_i18n?.[language] ?? item.name;
  const localizedDescription = (item: Template) => item.description_i18n?.[language] ?? item.description;

  const edit = (template: Template) => {
    setEditing(template);
    setCloning(undefined);
    form.setFieldsValue({ name: template.name, description: template.description, visibility: template.visibility });
    setOpen(true);
  };
  const canMutate = (template: Template) => Boolean(
    !template.builtin && me.data && (me.data.role === 'administrator' || (template.owner_id && me.data.id === template.owner_id)),
  );
  const openExisting = (match: TemplateDuplicateMatch) => {
    setPendingDuplicate(undefined);
    setOpen(false);
    navigate(`/experiments/new?template=${encodeURIComponent(match.id)}`);
  };

  return (
    <div className="page">
      <PageIntro title={t('templates.title')} subtitle={t('templates.subtitle')} actions={<Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/experiments/new')}>{t('templates.create')}</Button>} />
      <Card>
        <div className="template-filter-toolbar">
          <Input className="table-search" allowClear prefix={<SearchOutlined />} placeholder={t('common.search')} value={search} onChange={(event) => setFilter('q', event.target.value)} />
          <div className="template-filter-controls">
            <Select value={source} onChange={(value) => setFilter('source', value)} aria-label={t('templates.source')} options={[
              { value: 'all', label: t('templates.allSources') },
              { value: 'builtin', label: t('templates.builtinSource') },
              { value: 'mine', label: t('templates.mine') },
              { value: 'shared', label: t('templates.sharedByOthers') },
            ]} />
            <Select value={visibility} onChange={(value) => setFilter('visibility', value)} aria-label={t('templates.visibility')} options={[
              { value: 'all', label: t('templates.allVisibilities') },
              { value: 'personal', label: t('templates.personal') },
              { value: 'shared', label: t('templates.shared') },
            ]} />
            <Select value={architecture} onChange={(value) => setFilter('backbone', value)} aria-label={t('templates.backbone')} options={[{ value: 'all', label: t('templates.allBackbones') }, ...options.architectures.map((value) => ({ value, label: value }))]} />
            <Select value={method} onChange={(value) => setFilter('method', value)} aria-label={t('templates.method')} options={[{ value: 'all', label: t('templates.allMethods') }, ...options.methods.map((value) => ({ value, label: value }))]} />
            <Select value={dataset} onChange={(value) => setFilter('dataset', value)} aria-label={t('templates.dataset')} options={[{ value: 'all', label: t('templates.allDatasets') }, ...options.datasets.map((value) => ({ value, label: value }))]} />
            <Select value={availability} onChange={(value) => setFilter('availability', value)} aria-label={t('templates.availabilityFilter')} options={[
              { value: 'all', label: t('templates.allAvailabilities') },
              { value: 'ready', label: t('templates.availability.ready') },
              { value: 'partial', label: t('templates.availability.partial') },
              { value: 'missing', label: t('templates.availability.missing') },
              { value: 'custom', label: t('templates.customTemplate') },
            ]} />
            <Select value={category} onChange={(value) => setFilter('category', value)} aria-label={t('templates.category')} options={[{ value: 'all', label: t('templates.allCategories') }, ...options.categories.map((value) => ({ value, label: value }))]} />
          </div>
          <div className="template-filter-summary">
            <Typography.Text type="secondary">{t('templates.resultCount', { count: rows.length, total: allRows.length })}</Typography.Text>
            <Button type="link" size="small" disabled={!hasFilters} onClick={clearFilters}>{t('templates.clearFilters')}</Button>
          </div>
        </div>
        <AsyncState loading={templates.isLoading} error={templates.error} onRetry={() => void templates.refetch()}>
          {!allRows.length ? <Empty description={t('common.noData')} /> : <Table<Template>
            rowKey="id"
            dataSource={rows}
            locale={{ emptyText: t('templates.noFilterResults') }}
            pagination={{
              current: Math.min(page, maxPage),
              pageSize,
              total: rows.length,
              showSizeChanger: true,
              pageSizeOptions: PAGE_SIZES.map(String),
              showTotal: (total) => t('templates.paginationTotal', { total }),
              onChange: (nextPage, nextPageSize) => {
                const next = new URLSearchParams(searchParams);
                if (nextPageSize === PAGE_SIZES[0]) next.delete('page_size');
                else next.set('page_size', String(nextPageSize));
                if (nextPage === 1 || nextPageSize !== pageSize) next.delete('page');
                else next.set('page', String(nextPage));
                setSearchParams(next, { replace: true });
              },
            }}
            columns={[
              { title: t('templates.name'), dataIndex: 'name', render: (_: string, row) => <div><Space wrap><Typography.Text strong>{localizedName(row)}</Typography.Text>{row.builtin ? <Tag color="geekblue">{t('templates.builtin')}</Tag> : null}{row.availability ? <Tag color={row.availability === 'ready' ? 'green' : row.availability === 'partial' ? 'gold' : 'red'}>{t(`templates.availability.${row.availability}`)}</Tag> : null}</Space><small className="table-subtitle">{localizedDescription(row)}</small><Space size={[4, 4]} wrap>{row.tags?.map((tag) => <Tag key={tag}>{tag}</Tag>)}</Space></div> },
              { title: t('templates.visibility'), dataIndex: 'visibility', render: (value, row) => <Tag color={row.builtin ? 'geekblue' : value === 'shared' ? 'blue' : 'default'}>{row.builtin ? t('templates.builtin') : t(`templates.${value}`)}</Tag> },
              { title: t('builder.backbone'), dataIndex: 'architecture', responsive: ['md'] },
              { title: t('builder.method'), dataIndex: 'method', responsive: ['lg'] },
              { title: t('common.owner'), dataIndex: 'owner_name', responsive: ['md'] },
              { title: t('templates.version'), dataIndex: 'version', render: (value?: number) => `v${value ?? 1}`, responsive: ['lg'] },
              { title: t('common.updatedAt'), dataIndex: 'updated_at', render: (value?: string) => value ? dayjs(value).format('MM-DD HH:mm') : '—', responsive: ['xl'] },
              {
                title: t('common.actions'), width: 230,
                render: (_, row) => (
                  <Space size="small">
                    <Button type="primary" ghost size="small" icon={<RocketOutlined />} onClick={() => navigate(`/experiments/new?template=${encodeURIComponent(row.id)}`)}>{t('templates.use')}</Button>
                    {canMutate(row) ? <Button type="text" size="small" aria-label={t('common.edit')} icon={<EditOutlined />} onClick={() => edit(row)} /> : null}
                    <Button type="text" size="small" aria-label={t('common.copy')} icon={<CopyOutlined />} onClick={() => { setEditing(undefined); setCloning(row); form.setFieldsValue({ name: t('builder.copyName', { name: localizedName(row) }), description: localizedDescription(row), visibility: 'personal' }); setOpen(true); }} />
                    {canMutate(row) ? <Button danger type="text" size="small" aria-label={t('common.delete')} icon={<DeleteOutlined />} onClick={() => Modal.confirm({ title: t('common.delete'), content: row.name, okButtonProps: { danger: true }, onOk: () => remove.mutate(row.id) })} /> : null}
                  </Space>
                ),
              },
            ]}
          />}
        </AsyncState>
      </Card>
      <Modal
        title={editing ? t('common.edit') : t('templates.create')}
        open={open}
        onCancel={() => setOpen(false)}
        okText={t('common.save')}
        confirmLoading={save.isPending}
        onOk={() => form.submit()}
      >
        <Form<TemplateForm> form={form} layout="vertical" initialValues={{ visibility: 'personal' }} onFinish={(values) => save.mutate({ values, allowDuplicate: false, editingId: editing?.id, spec: cloning?.spec })}>
          <Form.Item name="name" label={t('templates.name')} rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="description" label={t('builder.description')}><Input.TextArea rows={3} /></Form.Item>
          <Form.Item name="visibility" label={t('templates.visibility')} rules={[{ required: true }]}><Radio.Group options={[{ label: t('templates.personal'), value: 'personal' }, { label: t('templates.shared'), value: 'shared' }]} /></Form.Item>
        </Form>
      </Modal>
      <TemplateDuplicateModal
        detail={pendingDuplicate?.detail}
        saving={save.isPending}
        onCancel={() => setPendingDuplicate(undefined)}
        onOpenExisting={openExisting}
        onSaveAnyway={() => pendingDuplicate && save.mutate({ ...pendingDuplicate.request, allowDuplicate: true })}
      />
    </div>
  );
}
