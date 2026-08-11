import { BulbOutlined, LockOutlined, UserOutlined } from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Button, Card, Form, Input, Segmented, Typography, message } from 'antd';
import { useTranslation } from 'react-i18next';
import { useLocation, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { Language } from '../api/types';
import { usePreferences } from '../app-context';

export function LoginPage() {
  const { t } = useTranslation();
  const { language, setLanguage } = usePreferences();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const login = useMutation({
    mutationFn: (values: { username: string; password: string }) => api.auth.login(values.username, values.password),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['me'] });
      const from = (location.state as { from?: string } | null)?.from ?? '/';
      navigate(from, { replace: true });
    },
    onError: (error) => message.error(error instanceof Error ? error.message : String(error)),
  });
  return (
    <div className="auth-layout login-layout">
      <div className="auth-language">
        <Segmented
          value={language}
          options={[{ label: '简体中文', value: 'zh-CN' }, { label: 'English', value: 'en-US' }]}
          onChange={(value) => setLanguage(value as Language)}
        />
      </div>
      <Card className="login-card" bordered={false}>
        <div className="login-brand"><span className="auth-logo"><BulbOutlined /></span></div>
        <Typography.Title level={2}>{t('login.title')}</Typography.Title>
        <Typography.Paragraph type="secondary">{t('login.subtitle')}</Typography.Paragraph>
        <Form layout="vertical" size="large" onFinish={(values) => login.mutate(values as { username: string; password: string })}>
          <Form.Item name="username" label={t('login.username')} rules={[{ required: true }]}>
            <Input prefix={<UserOutlined />} autoFocus autoComplete="username" />
          </Form.Item>
          <Form.Item name="password" label={t('login.password')} rules={[{ required: true }]}>
            <Input.Password prefix={<LockOutlined />} autoComplete="current-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block loading={login.isPending}>{t('login.submit')}</Button>
        </Form>
      </Card>
      <Typography.Text type="secondary" className="auth-footer">{t('login.footer')}</Typography.Text>
    </div>
  );
}
