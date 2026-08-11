import { SearchOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Card, Input, Typography } from 'antd';
import dayjs from 'dayjs';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api, listFrom } from '../api/client';
import type { AuditEvent } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';

export function AuditPage() {
  const { t } = useTranslation();
  const [search, setSearch] = useState('');
  const query = useQuery({ queryKey: ['audit'], queryFn: () => api.audit.list(500), refetchInterval: 15_000 });
  const rows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return listFrom(query.data).filter((item) => !term || `${item.action} ${item.target_type ?? ''} ${item.target_id ?? ''} ${item.actor_id ?? ''}`.toLowerCase().includes(term));
  }, [query.data, search]);
  return (
    <div className="page">
      <PageIntro title={t('audit.title')} subtitle={t('audit.subtitle')} />
      <Card>
        <div className="table-toolbar"><Input className="table-search" allowClear prefix={<SearchOutlined />} placeholder={t('common.search')} value={search} onChange={(event) => setSearch(event.target.value)} /></div>
        <AsyncState loading={query.isLoading} error={query.error} empty={!rows.length} onRetry={() => void query.refetch()}>
          <Table<AuditEvent>
            rowKey="id"
            dataSource={rows}
            expandable={{ expandedRowRender: (row) => <pre className="config-preview">{JSON.stringify(row.detail, null, 2)}</pre> }}
            columns={[
              { title: t('audit.action'), dataIndex: 'action', render: (value: string) => <Typography.Text code>{value}</Typography.Text> },
              { title: t('audit.target'), render: (_, row) => `${row.target_type || '—'}${row.target_id ? ` · ${row.target_id}` : ''}` },
              { title: t('audit.actor'), dataIndex: 'actor_id', render: (value?: string) => value || t('audit.system') },
              { title: t('common.createdAt'), dataIndex: 'created_at', render: (value: string) => dayjs(value).format('YYYY-MM-DD HH:mm:ss') },
            ]}
          />
        </AsyncState>
      </Card>
    </div>
  );
}
