import { PlusOutlined, RocketOutlined, SearchOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Button, Card, Input, Select, Space, Tag, Typography } from 'antd';
import dayjs from 'dayjs';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { api, listFrom } from '../api/client';
import type { Experiment, Template } from '../api/types';
import { usePreferences } from '../app-context';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';

export function ExperimentsPage() {
  const { t } = useTranslation();
  const { language } = usePreferences();
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState<string>();
  const query = useQuery({ queryKey: ['experiments'], queryFn: api.experiments.list, refetchInterval: 15_000 });
  const templates = useQuery({ queryKey: ['templates'], queryFn: api.templates.list });
  const presets = useMemo(() => listFrom(templates.data).filter((item) => item.builtin).slice(0, 5), [templates.data]);
  const presetName = (item: Template) => item.name_i18n?.[language] ?? item.name;
  const presetDescription = (item: Template) => item.description_i18n?.[language] ?? item.description;
  const rows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return listFrom(query.data).filter((item) => (
      (!status || item.status === status)
      && (!term || `${item.name} ${item.owner_name ?? ''} ${item.architecture ?? ''} ${item.method ?? ''} ${item.dataset ?? ''}`.toLowerCase().includes(term))
    ));
  }, [query.data, search, status]);
  return (
    <div className="page">
      <PageIntro
        title={t('experiments.title')}
        subtitle={t('experiments.subtitle')}
        actions={<Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/experiments/new')}>{t('nav.newExperiment')}</Button>}
      />
      {presets.length ? (
        <Card className="preset-section" title={t('experiments.quickStarts')} extra={<Button type="link" onClick={() => navigate('/templates')}>{t('experiments.viewAllTemplates')}</Button>}>
          <div className="preset-card-grid">
            {presets.map((preset) => (
              <Card key={preset.id} size="small" className="preset-card">
                <Space wrap><Tag color="geekblue">{t('templates.builtin')}</Tag><Tag color={preset.availability === 'ready' ? 'green' : preset.availability === 'partial' ? 'gold' : 'red'}>{t(`templates.availability.${preset.availability ?? 'missing'}`)}</Tag></Space>
                <Typography.Title level={5}>{presetName(preset)}</Typography.Title>
                <Typography.Paragraph type="secondary">{presetDescription(preset)}</Typography.Paragraph>
                <Space size={[4, 4]} wrap>{preset.tags?.slice(0, 3).map((tag) => <Tag key={tag}>{tag}</Tag>)}</Space>
                <Button block type="primary" ghost icon={<RocketOutlined />} onClick={() => navigate(`/experiments/new?template=${encodeURIComponent(preset.id)}`)}>{t('templates.use')}</Button>
              </Card>
            ))}
          </div>
        </Card>
      ) : null}
      <Card>
        <div className="table-toolbar">
          <Space wrap>
            <Input allowClear prefix={<SearchOutlined />} placeholder={t('common.search')} value={search} onChange={(event) => setSearch(event.target.value)} />
            <Select allowClear placeholder={t('common.status')} value={status} onChange={setStatus} options={['queued', 'running', 'completed', 'failed', 'stopped'].map((value) => ({ value, label: t(`status.${value}`) }))} />
          </Space>
        </div>
        <AsyncState loading={query.isLoading} error={query.error} empty={!rows.length} onRetry={() => void query.refetch()}>
          <Table<Experiment>
            rowKey="id"
            dataSource={rows}
            onRow={(row) => ({ onClick: () => navigate(`/experiments/${row.id}`), style: { cursor: 'pointer' } })}
            columns={[
              { title: t('experiments.name'), dataIndex: 'name', render: (value: string, row) => <div><Typography.Text strong>{value}</Typography.Text><small className="table-subtitle">{row.owner_name}</small></div> },
              { title: t('common.status'), dataIndex: 'status', render: (value: string) => <Tag color={value === 'completed' ? 'green' : value === 'failed' ? 'red' : value === 'running' ? 'blue' : 'default'}>{t(`status.${value}`, { defaultValue: value })}</Tag> },
              { title: t('experiments.architecture'), dataIndex: 'architecture', render: (value?: string) => value || '—' },
              { title: t('experiments.method'), dataIndex: 'method', render: (value?: string) => value || '—' },
              { title: t('experiments.dataset'), dataIndex: 'dataset', render: (value?: string) => value || '—' },
              { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm') : '—' },
            ]}
          />
        </AsyncState>
      </Card>
    </div>
  );
}
