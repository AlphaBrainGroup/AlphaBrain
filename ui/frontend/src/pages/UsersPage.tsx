import { DeleteOutlined, EditOutlined, PlusOutlined, SearchOutlined, TeamOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Button, Card, Form, Input, Modal, Radio, Result, Space, Switch, Tag, Typography, message } from 'antd';
import dayjs from 'dayjs';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api, listFrom } from '../api/client';
import type { User, UserRole } from '../api/types';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ResizableTable as Table } from '../components/ResizableTable';

interface UserForm {
  username: string;
  display_name?: string;
  role: UserRole;
  password?: string;
  active: boolean;
}

export function UsersPage() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<User>();
  const [search, setSearch] = useState('');
  const [form] = Form.useForm<UserForm>();
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const users = useQuery({ queryKey: ['users'], queryFn: api.users.list, enabled: me.data?.role === 'administrator' });
  const save = useMutation({
    mutationFn: (values: UserForm) => editing ? api.users.update(editing.id, values) : api.users.create(values),
    onSuccess: async () => { setOpen(false); form.resetFields(); await queryClient.invalidateQueries({ queryKey: ['users'] }); },
    onError: (error) => message.error(error.message),
  });
  const remove = useMutation({
    mutationFn: api.users.remove,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['users'] }),
    onError: (error) => message.error(error.message),
  });
  const rows = useMemo(() => {
    const term = search.trim().toLowerCase();
    return listFrom(users.data).filter((user) => !term || `${user.username} ${user.display_name ?? ''}`.toLowerCase().includes(term));
  }, [search, users.data]);

  const edit = (user: User) => {
    setEditing(user);
    form.setFieldsValue({ username: user.username, display_name: user.display_name, role: user.role, active: user.active !== false });
    setOpen(true);
  };

  if (!me.isLoading && me.data?.role !== 'administrator') return <Result status="403" title="403" subTitle={t('users.adminRequired')} />;
  return (
    <div className="page">
      <PageIntro title={t('users.title')} subtitle={t('users.subtitle')} actions={<Button type="primary" icon={<PlusOutlined />} onClick={() => { setEditing(undefined); form.resetFields(); form.setFieldsValue({ role: 'researcher', active: true }); setOpen(true); }}>{t('users.add')}</Button>} />
      <Card>
        <div className="table-toolbar"><Input className="table-search" prefix={<SearchOutlined />} allowClear placeholder={t('common.search')} value={search} onChange={(event) => setSearch(event.target.value)} /></div>
        <AsyncState loading={me.isLoading || users.isLoading} error={me.error ?? users.error} empty={!rows.length} onRetry={() => void users.refetch()}>
          <Table<User>
            rowKey="id"
            dataSource={rows}
            columns={[
              { title: t('users.username'), dataIndex: 'username', render: (value: string, row) => <Space><span className="user-table-avatar"><TeamOutlined /></span><div><Typography.Text strong>{row.display_name || value}</Typography.Text><small className="table-subtitle">@{value}</small></div></Space> },
              { title: t('users.role'), dataIndex: 'role', render: (value: UserRole) => <Tag color={value === 'administrator' ? 'purple' : 'blue'}>{t(`users.${value}`)}</Tag> },
              { title: t('users.active'), dataIndex: 'active', render: (value?: boolean) => <Tag color={value === false ? 'default' : 'green'}>{value === false ? t('common.disabled') : t('common.enabled')}</Tag> },
              { title: t('common.createdAt'), dataIndex: 'created_at', render: (value?: string) => value ? dayjs(value).format('YYYY-MM-DD HH:mm') : '—', responsive: ['md'] },
              { title: t('common.actions'), width: 130, render: (_, row) => <Space><Button type="text" icon={<EditOutlined />} onClick={() => edit(row)} /><Button type="text" danger disabled={row.id === me.data?.id} icon={<DeleteOutlined />} onClick={() => Modal.confirm({ title: t('common.delete'), content: row.username, okButtonProps: { danger: true }, onOk: () => remove.mutate(row.id) })} /></Space> },
            ]}
          />
        </AsyncState>
      </Card>
      <Modal title={editing ? t('common.edit') : t('users.add')} open={open} onCancel={() => setOpen(false)} okText={t('common.save')} confirmLoading={save.isPending} onOk={() => form.submit()}>
        <Form<UserForm> form={form} layout="vertical" initialValues={{ role: 'researcher', active: true }} onFinish={(values) => save.mutate(values)}>
          <Form.Item name="username" label={t('users.username')} rules={[{ required: true }]}><Input disabled={Boolean(editing)} autoComplete="off" /></Form.Item>
          <Form.Item name="display_name" label={t('users.displayName')}><Input /></Form.Item>
          <Form.Item name="role" label={t('users.role')} rules={[{ required: true }]}><Radio.Group options={[{ label: t('users.researcher'), value: 'researcher' }, { label: t('users.administrator'), value: 'administrator' }]} /></Form.Item>
          <Form.Item name="password" label={t('users.password')} rules={editing ? [] : [{ required: true, min: 8 }]}><Input.Password autoComplete="new-password" /></Form.Item>
          <Form.Item name="active" label={t('users.active')} valuePropName="checked"><Switch /></Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
