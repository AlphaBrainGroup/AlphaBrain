import { EyeOutlined, PlusOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Button, Card, Input, Segmented, Space, Typography } from 'antd';
import dayjs from 'dayjs';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { api, listFrom } from '../api/client';
import type { Job, JobStatus } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { StatusTag } from '../components/StatusTag';

function duration(job: Job): string {
  if (!job.started_at) return '—';
  const end = job.finished_at ? dayjs(job.finished_at) : dayjs();
  const seconds = Math.max(0, end.diff(dayjs(job.started_at), 'second'));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m ${seconds % 60}s`;
}

export function JobsPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [filter, setFilter] = useState<'all' | JobStatus>('all');
  const [search, setSearch] = useState('');
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: () => api.jobs.list(), refetchInterval: 5_000 });
  const rows = useMemo(() => listFrom(jobs.data).filter((job) => {
    const matchesStatus = filter === 'all' || job.status === filter;
    const term = search.trim().toLowerCase();
    const matchesSearch = !term || `${job.name} ${job.owner_name ?? ''} ${job.id}`.toLowerCase().includes(term);
    return matchesStatus && matchesSearch;
  }), [filter, jobs.data, search]);

  return (
    <div className="page">
      <PageIntro
        title={t('jobs.title')}
        subtitle={t('jobs.subtitle')}
        actions={<Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/experiments/new')}>{t('dashboard.newExperiment')}</Button>}
      />
      <Card>
        <div className="table-toolbar">
          <Segmented
            value={filter}
            onChange={(value) => setFilter(value as 'all' | JobStatus)}
            options={[
              { label: t('common.all'), value: 'all' },
              { label: t('status.running'), value: 'running' },
              { label: t('status.queued'), value: 'queued' },
              { label: t('status.completed'), value: 'completed' },
              { label: t('status.failed'), value: 'failed' },
            ]}
          />
          <Space>
            <Input allowClear prefix={<SearchOutlined />} placeholder={t('common.search')} value={search} onChange={(event) => setSearch(event.target.value)} />
            <Button icon={<ReloadOutlined />} onClick={() => void jobs.refetch()}>{t('common.refresh')}</Button>
          </Space>
        </div>
        <AsyncState loading={jobs.isLoading} error={jobs.error} empty={!rows.length} onRetry={() => void jobs.refetch()}>
          <Table<Job>
            rowKey="id"
            dataSource={rows}
            pagination={{ pageSize: 15, showSizeChanger: true }}
            onRow={(record) => ({ onClick: () => navigate(`/jobs/${record.id}`) })}
            columns={[
              {
                title: t('jobs.name'), dataIndex: 'name',
                render: (value: string, row) => <div><Typography.Text strong>{value}</Typography.Text><small className="table-subtitle">{row.stage_name || row.id.slice(0, 8)}</small></div>,
              },
              { title: t('common.status'), dataIndex: 'status', render: (value) => <StatusTag status={value} /> },
              { title: t('common.owner'), dataIndex: 'owner_name', responsive: ['md'] },
              { title: t('jobs.gpus'), dataIndex: 'gpu_ids', render: (values?: number[]) => values?.length ? values.map((value) => <span className="gpu-chip" key={value}>{value}</span>) : '—' },
              { title: t('jobs.queue'), dataIndex: 'queue_position', render: (value?: number) => value ? `#${value}` : '—', responsive: ['lg'] },
              { title: t('jobs.duration'), render: (_, row) => duration(row), responsive: ['lg'] },
              { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('MM-DD HH:mm') : '—', responsive: ['xl'] },
              { title: t('common.actions'), width: 90, render: (_, row) => <Button type="link" icon={<EyeOutlined />} onClick={(event) => { event.stopPropagation(); navigate(`/jobs/${row.id}`); }}>{t('common.view')}</Button> },
            ]}
          />
        </AsyncState>
      </Card>
    </div>
  );
}
