import { DeleteOutlined, DownloadOutlined, FileZipOutlined, SearchOutlined } from '@ant-design/icons';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Button, Card, Input, Modal, Select, Space, Tag, Typography, message } from 'antd';
import dayjs from 'dayjs';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { api, listFrom } from '../api/client';
import type { Workload } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';
import { UtilityProgress } from '../components/UtilityProgress';

const kinds: Workload['kind'][] = ['training', 'deployment', 'evaluation', 'utility'];

export function WorkloadsPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [kind, setKind] = useState<Workload['kind']>();
  const [status, setStatus] = useState<string>();
  const query = useQuery({ queryKey: ['workloads'], queryFn: () => api.workloads.list(), refetchInterval: 5_000 });
  const removeTraining = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => api.jobs.remove(id, name),
    onSuccess: async () => {
      message.success(t('workloads.trainingDeleted'));
      await query.refetch();
    },
    onError: (error) => message.error(error.message),
  });
  const packageTraining = useMutation({
    mutationFn: (id: string) => api.jobs.package(id),
    onSuccess: async () => {
      message.success(t('workloads.packageQueued'));
      await query.refetch();
    },
    onError: (error) => message.error(error.message),
  });
  const cancelPackage = useMutation({
    mutationFn: (id: string) => api.utilities.cancel(id),
    onSuccess: async () => {
      message.success(t('workloads.packageCancelled'));
      await query.refetch();
    },
    onError: (error) => message.error(error.message),
  });
  const rows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return listFrom(query.data).filter((item) => (
      (!kind || item.kind === kind)
      && (!status || item.status === status)
      && (!term || `${item.name} ${item.owner_name ?? ''} ${item.subtype ?? ''}`.toLowerCase().includes(term))
    ));
  }, [kind, query.data, search, status]);
  const packageControl = (row: Workload) => {
    const run = row.package_run;
    if (run && ['queued', 'starting', 'running', 'stopping'].includes(run.status)) {
      return (
        <Space>
          <UtilityProgress run={run} compact />
          <Button danger size="small" loading={cancelPackage.isPending && cancelPackage.variables === run.id} onClick={(event) => { event.stopPropagation(); cancelPackage.mutate(run.id); }}>{t('common.cancel')}</Button>
        </Space>
      );
    }
    return (
      <Space>
        {run?.status === 'completed' ? <Button size="small" icon={<DownloadOutlined />} href={api.utilities.outputUrl(run.id)} onClick={(event) => event.stopPropagation()}>{t('common.download')}</Button> : null}
        <Button size="small" icon={<FileZipOutlined />} disabled={!row.can_package} loading={packageTraining.isPending && packageTraining.variables === row.id} onClick={(event) => { event.stopPropagation(); packageTraining.mutate(row.id); }}>
          {run?.status === 'completed' ? t('workloads.repackage') : t('workloads.packageTraining')}
        </Button>
      </Space>
    );
  };
  return (
    <div className="page">
      <PageIntro title={t('workloads.title')} subtitle={t('workloads.subtitle')} />
      <Card>
        <div className="table-toolbar">
          <Space wrap>
            <Input allowClear prefix={<SearchOutlined />} placeholder={t('common.search')} value={search} onChange={(event) => setSearch(event.target.value)} />
            <Select allowClear placeholder={t('workloads.kind')} value={kind} onChange={setKind} options={kinds.map((value) => ({ value, label: t(`workloads.kinds.${value}`) }))} />
            <Select allowClear placeholder={t('common.status')} value={status} onChange={setStatus} options={['queued', 'starting', 'running', 'stopping', 'completed', 'failed', 'stopped', 'cancelled'].map((value) => ({ value, label: t(`status.${value}`) }))} />
          </Space>
        </div>
        <AsyncState loading={query.isLoading} error={query.error} empty={!rows.length} onRetry={() => void query.refetch()}>
          <Table<Workload>
            rowKey={(row) => `${row.kind}:${row.id}`}
            dataSource={rows}
            onRow={(row) => ({ onClick: () => navigate(row.detail_url), style: { cursor: 'pointer' } })}
            scroll={{ x: 1300 }}
            columns={[
              { title: t('workloads.workload'), dataIndex: 'name', fixed: 'left', render: (value: string, row) => <div><Typography.Text strong>{value}</Typography.Text><small className="table-subtitle">{t(`workloads.kinds.${row.kind}`)}{row.subtype ? ` · ${row.subtype}` : ''}</small></div> },
              { title: t('common.status'), dataIndex: 'status', render: (value: string) => <Tag color={value === 'running' ? 'blue' : value === 'completed' ? 'green' : value === 'failed' ? 'red' : 'default'}>{t(`status.${value}`, { defaultValue: value })}</Tag> },
              { title: t('common.owner'), dataIndex: 'owner_name', render: (value?: string) => value || '—' },
              { title: t('workloads.queue'), render: (_, row) => <div>{t(`workloads.scopes.${row.queue_scope}`)}<small className="table-subtitle">{row.queue_position ? `#${row.queue_position}` : '—'}</small></div> },
              { title: 'GPU', dataIndex: 'gpu_ids', render: (value: number[]) => value.length ? value.map((id) => <Tag key={id}>GPU {id}</Tag>) : '—' },
              { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm') : '—' },
              { title: t('workloads.error'), dataIndex: 'error', render: (value?: string) => value || '—' },
              { title: t('workloads.packageTraining'), width: 290, render: (_, row) => row.kind === 'training' ? packageControl(row) : null },
              {
                title: t('common.actions'),
                fixed: 'right',
                render: (_, row) => row.kind === 'training' && row.can_delete ? (
                  <Button
                    danger
                    size="small"
                    icon={<DeleteOutlined />}
                    loading={removeTraining.isPending && removeTraining.variables?.id === row.id}
                    onClick={(event) => {
                      event.stopPropagation();
                      Modal.confirm({
                        title: t('workloads.deleteTrainingTitle'),
                        content: t('workloads.deleteTrainingConfirm', { name: row.name }),
                        okText: t('common.delete'),
                        cancelText: t('common.cancel'),
                        okButtonProps: { danger: true },
                        onOk: () => removeTraining.mutateAsync({ id: row.id, name: row.name }),
                      });
                    }}
                  >
                    {t('workloads.deleteTraining')}
                  </Button>
                ) : null,
              },
            ]}
          />
        </AsyncState>
      </Card>
    </div>
  );
}
