import { BulbOutlined, CheckCircleFilled, GlobalOutlined } from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Form, Input, Radio, Segmented, Space, Typography, message } from 'antd';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { Language } from '../api/types';
import { usePreferences } from '../app-context';

interface SetupForm {
  mode: 'personal' | 'laboratory';
  username: string;
  display_name: string;
  password?: string;
  confirm_password?: string;
}

export function SetupPage({ initialMode = 'personal' }: { initialMode?: 'personal' | 'laboratory' }) {
  const { t } = useTranslation();
  const { language, setLanguage } = usePreferences();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<'personal' | 'laboratory'>(initialMode);
  const [form] = Form.useForm<SetupForm>();
  const setup = useMutation({
    mutationFn: (values: SetupForm) => api.setup.create({
      mode: values.mode,
      username: values.username || 'admin',
      display_name: values.display_name || 'Administrator',
      password: values.mode === 'laboratory' ? values.password : undefined,
      locale: language,
    }),
    onSuccess: async (result, values) => {
      queryClient.setQueryData(['setup-status'], { configured: true, mode: values.mode });
      queryClient.setQueryData(['me'], result.user);
      if (language !== 'zh-CN') await api.settings.updatePreferences({ locale: language }).catch(() => undefined);
      await queryClient.invalidateQueries({ queryKey: ['setup-status'], refetchType: 'none' });
      navigate('/', { replace: true });
    },
    onError: (error) => message.error(error instanceof Error ? error.message : String(error)),
  });

  return (
    <div className="auth-layout setup-layout">
      <div className="auth-language">
        <Segmented
          value={language}
          options={[{ label: '简体中文', value: 'zh-CN' }, { label: 'English', value: 'en-US' }]}
          onChange={(value) => setLanguage(value as Language)}
        />
      </div>
      <div className="setup-grid">
        <section className="setup-hero">
          <span className="auth-logo"><BulbOutlined /></span>
          <Typography.Title>AlphaBrain</Typography.Title>
          <Typography.Title level={2}>{t('setup.title')}</Typography.Title>
          <Typography.Paragraph>{t('setup.subtitle')}</Typography.Paragraph>
          <div className="setup-points">
            <span><CheckCircleFilled /> {t('setup.visualComposition')}</span>
            <span><CheckCircleFilled /> {t('setup.gpuScheduling')}</span>
            <span><CheckCircleFilled /> {t('setup.liveMonitoring')}</span>
          </div>
        </section>
        <Card className="setup-card" bordered={false}>
          <Form<SetupForm>
            form={form}
            layout="vertical"
            initialValues={{ mode: initialMode, username: 'admin', display_name: 'Administrator' }}
            onFinish={(values) => setup.mutate(values)}
          >
            <Form.Item name="mode" label={t('settings.mode')}>
              <Radio.Group
                className="mode-picker"
                onChange={(event) => setMode(event.target.value as 'personal' | 'laboratory')}
              >
                <Radio.Button value="personal">
                  <b>{t('setup.personal')}</b><small>{t('setup.personalDesc')}</small>
                </Radio.Button>
                <Radio.Button value="laboratory">
                  <b>{t('setup.laboratory')}</b><small>{t('setup.laboratoryDesc')}</small>
                </Radio.Button>
              </Radio.Group>
            </Form.Item>
            <Typography.Title level={4}>{t('setup.adminTitle')}</Typography.Title>
            <div className="form-grid-2">
              <Form.Item name="username" label={t('setup.username')} rules={[{ required: true }]}>
                <Input autoComplete="username" />
              </Form.Item>
              <Form.Item name="display_name" label={t('setup.displayName')} rules={[{ required: true }]}>
                <Input />
              </Form.Item>
            </div>
            {mode === 'laboratory' ? (
              <div className="form-grid-2">
                <Form.Item name="password" label={t('setup.password')} rules={[{ required: true, min: 8 }]}>
                  <Input.Password autoComplete="new-password" />
                </Form.Item>
                <Form.Item
                  name="confirm_password"
                  label={t('setup.confirmPassword')}
                  dependencies={['password']}
                  rules={[
                    { required: true },
                    ({ getFieldValue }) => ({
                      validator(_, value) {
                        return !value || getFieldValue('password') === value
                          ? Promise.resolve()
                          : Promise.reject(new Error(t('setup.passwordMismatch')));
                      },
                    }),
                  ]}
                >
                  <Input.Password autoComplete="new-password" />
                </Form.Item>
              </div>
            ) : (
              <Alert type="info" showIcon message={t('setup.personalDesc')} />
            )}
            <Button type="primary" htmlType="submit" size="large" block loading={setup.isPending}>
              {t('setup.finish')}
            </Button>
          </Form>
          <Space className="setup-footnote"><GlobalOutlined /> {language === 'zh-CN' ? '默认语言：简体中文' : 'Default language: English'}</Space>
        </Card>
      </div>
    </div>
  );
}
