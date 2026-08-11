import { CloudServerOutlined, CodeOutlined, DatabaseOutlined, DeleteOutlined, ExperimentOutlined, FolderOpenOutlined, GlobalOutlined, KeyOutlined, SafetyCertificateOutlined, SaveOutlined, SettingOutlined, ToolOutlined } from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Collapse, Form, Input, InputNumber, Modal, Radio, Select, Space, Switch, Tabs, Tag, Typography, message } from 'antd';
import { useEffect, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import type { Language, SystemSettings, ThemeMode, User } from '../api/types';
import { usePreferences } from '../app-context';
import { AsyncState } from '../components/AsyncState';
import { PageIntro } from '../components/PageIntro';
import { ServerDirectoryPicker } from '../components/ServerDirectoryPicker';

function SettingsGroupLabel({ icon, title, description, status }: {
  icon: ReactNode;
  title: string;
  description: string;
  status?: ReactNode;
}) {
  return (
    <div className="settings-group-label">
      <span className="settings-group-icon">{icon}</span>
      <span className="settings-group-copy">
        <Typography.Text strong>{title}</Typography.Text>
        <Typography.Text type="secondary">{description}</Typography.Text>
      </span>
      {status ? <span className="settings-group-status">{status}</span> : null}
    </div>
  );
}

export function SettingsPage() {
  const { t } = useTranslation();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const { language, setLanguage, theme, setTheme } = usePreferences();
  const queryClient = useQueryClient();
  const [generalForm] = Form.useForm<SystemSettings>();
  const [environmentForm] = Form.useForm<SystemSettings>();
  const [experimentalForm] = Form.useForm<SystemSettings>();
  const [preferenceForm] = Form.useForm<Partial<User>>();
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me });
  const isAdmin = me.data?.role === 'administrator';
  const settings = useQuery({ queryKey: ['settings'], queryFn: api.settings.get, enabled: isAdmin });
  const wandbStatus = useQuery({
    queryKey: ['settings', 'wandb'],
    queryFn: api.settings.wandb.status,
    enabled: isAdmin,
  });
  const [wandbApiKey, setWandbApiKey] = useState('');
  const [wandbSaving, setWandbSaving] = useState(false);
  const [wandbDeleting, setWandbDeleting] = useState(false);
  const selectedMode = Form.useWatch('mode', generalForm);
  const remoteTrainingEnabled = Form.useWatch('remote_training_enabled', environmentForm);

  useEffect(() => {
    if (settings.data) {
      generalForm.setFieldsValue(settings.data);
      environmentForm.setFieldsValue(settings.data);
      experimentalForm.setFieldsValue(settings.data);
    }
  }, [environmentForm, experimentalForm, generalForm, settings.data]);
  useEffect(() => {
    if (me.data) preferenceForm.setFieldsValue({
      locale: me.data.locale ?? language,
      theme: me.data.theme ?? theme,
      gpu_refresh_interval_seconds: me.data.gpu_refresh_interval_seconds ?? 5,
      experimental_enabled: me.data.experimental_enabled,
    });
  }, [language, me.data, preferenceForm, theme]);

  const updateSystem = useMutation({
    mutationFn: api.settings.update,
    onSuccess: async (_, variables) => {
      message.success(t('settings.saved'));
      if (variables.mode && variables.mode !== me.data?.deployment_mode) {
        queryClient.clear();
        navigate(variables.mode === 'laboratory' ? '/login' : '/', { replace: true });
        return;
      }
      await queryClient.invalidateQueries({ queryKey: ['settings'] });
    },
    onError: (error) => message.error(error.message),
  });
  const updatePreferences = useMutation({
    mutationFn: api.settings.updatePreferences,
    onSuccess: async (user) => {
      if (user.locale) setLanguage(user.locale);
      if (user.theme) setTheme(user.theme);
      message.success(t('settings.saved'));
      await queryClient.invalidateQueries({ queryKey: ['me'] });
    },
    onError: (error) => message.error(error.message),
  });
  const requestedTab = searchParams.get('tab') ?? 'general';
  const activeTab = requestedTab === 'experimental' ? 'general' : requestedTab;

  const savePreferences = (values: Partial<User>) => {
    const currentlyEnabled = Boolean(me.data?.experimental_enabled);
    if (!currentlyEnabled && values.experimental_enabled) {
      Modal.confirm({
        title: t('builder.experimental'),
        content: t('builder.experimentalRisk'),
        okButtonProps: { danger: true },
        onOk: () => updatePreferences.mutate(values),
      });
    } else updatePreferences.mutate(values);
  };

  const saveWandbApiKey = async () => {
    const apiKey = wandbApiKey.trim();
    if (apiKey.length < 8 || /\s/.test(apiKey)) {
      message.error(t('settings.wandbApiKeyInvalid'));
      return;
    }
    setWandbSaving(true);
    try {
      const status = await api.settings.wandb.setApiKey(apiKey);
      queryClient.setQueryData(['settings', 'wandb'], status);
      setWandbApiKey('');
      message.success(t(wandbStatus.data?.configured ? 'settings.wandbApiKeyReplaced' : 'settings.wandbApiKeySaved'));
    } catch (error) {
      message.error(error instanceof Error ? error.message : String(error));
    } finally {
      setWandbSaving(false);
    }
  };

  const confirmDeleteWandbApiKey = () => {
    Modal.confirm({
      title: t('settings.wandbDeleteTitle'),
      content: t('settings.wandbDeleteDescription'),
      okText: t('common.delete'),
      okButtonProps: { danger: true },
      onOk: async () => {
        setWandbDeleting(true);
        try {
          const status = await api.settings.wandb.deleteApiKey();
          queryClient.setQueryData(['settings', 'wandb'], status);
          setWandbApiKey('');
          message.success(t('settings.wandbApiKeyDeleted'));
        } catch (error) {
          message.error(error instanceof Error ? error.message : String(error));
          throw error;
        } finally {
          setWandbDeleting(false);
        }
      },
    });
  };

  return (
    <div className="page settings-page">
      <PageIntro title={t('settings.title')} subtitle={t('settings.subtitle')} />
      <AsyncState loading={me.isLoading || (isAdmin && settings.isLoading)} error={(isAdmin ? settings.error : undefined) ?? me.error} onRetry={() => { if (isAdmin) void settings.refetch(); void me.refetch(); }}>
        <Card className="settings-shell">
          <Tabs
            className="settings-tabs"
            tabPosition="left"
            activeKey={activeTab}
            onChange={(value) => setSearchParams({ tab: value })}
            items={[
              {
                key: 'general', label: <Space><SettingOutlined />{t('settings.general')}</Space>,
                children: (
                  <div className="settings-panel">
                    <div className="settings-panel-heading">
                      <Typography.Title level={4}>{t('settings.general')}</Typography.Title>
                      <Typography.Paragraph type="secondary">{t('settings.generalDescription')}</Typography.Paragraph>
                    </div>
                    <Card size="small" className="settings-section-card" title={t('settings.deploymentAndSecurity')}>
                      <Form<SystemSettings> form={generalForm} layout="vertical" disabled={!isAdmin} onFinish={(values) => updateSystem.mutate(values)}>
                        <Form.Item name="mode" label={t('settings.mode')}>
                          <Radio.Group className="mode-picker" optionType="button">
                            <Radio.Button value="personal"><b>{t('settings.personal')}</b><small>{t('settings.personalDesc')}</small></Radio.Button>
                            <Radio.Button value="laboratory"><b>{t('settings.laboratory')}</b><small>{t('settings.laboratoryDesc')}</small></Radio.Button>
                          </Radio.Group>
                        </Form.Item>
                        {selectedMode === 'laboratory' && settings.data?.mode === 'personal' ? (
                          <Form.Item name="admin_password" label={t('setup.password')} rules={[{ required: true, min: 8 }]}><Input.Password autoComplete="new-password" /></Form.Item>
                        ) : null}
                        <Form.Item name="secure_cookies" label={t('settings.secureCookies')} valuePropName="checked" extra={settings.data?.secure_cookies_locked ? t('settings.secureCookiesLocked') : t('settings.secureCookiesHint')}>
                          <Switch disabled={settings.data?.secure_cookies_locked} />
                        </Form.Item>
                        {!isAdmin ? <Alert type="info" showIcon message={t('settings.adminRequired')} /> : null}
                        {isAdmin ? (
                          <div className="settings-save-row">
                            <Button type="primary" htmlType="submit" icon={<SaveOutlined />} loading={updateSystem.isPending}>{t('common.save')}</Button>
                          </div>
                        ) : null}
                      </Form>
                    </Card>
                    <Collapse
                      className="settings-groups settings-advanced-group"
                      items={[{
                        key: 'experimental',
                        label: <SettingsGroupLabel icon={<ExperimentOutlined />} title={t('settings.experimental')} description={t('settings.experimentalSectionDescription')} />,
                        children: (
                          <>
                            <Alert type="warning" showIcon message={t('settings.allowExperimentalDesc')} description={t('builder.experimentalRisk')} />
                            <Form<SystemSettings> form={experimentalForm} layout="vertical" disabled={!isAdmin} onFinish={(values) => updateSystem.mutate(values)}>
                              <Form.Item name="experimental_allowed" label={t('settings.allowExperimental')} valuePropName="checked"><Switch /></Form.Item>
                              {isAdmin ? <Button htmlType="submit" icon={<SafetyCertificateOutlined />} loading={updateSystem.isPending}>{t('common.save')}</Button> : null}
                            </Form>
                          </>
                        ),
                      }]}
                    />
                  </div>
                ),
              },
              {
                key: 'environment', label: <Space><FolderOpenOutlined />{t('settings.infrastructure')}</Space>,
                children: (
                  <div className="settings-panel">
                    <div className="settings-panel-heading">
                      <Typography.Title level={4}>{t('settings.infrastructure')}</Typography.Title>
                      <Typography.Paragraph type="secondary">{t('settings.infrastructureDescription')}</Typography.Paragraph>
                    </div>
                    <Form<SystemSettings> form={environmentForm} layout="vertical" disabled={!isAdmin} onFinish={(values) => updateSystem.mutate(values)}>
                      <Collapse
                        className="settings-groups"
                        defaultActiveKey={remoteTrainingEnabled ? ['storage', 'remote'] : ['storage']}
                        items={[
                          {
                            key: 'storage',
                            label: <SettingsGroupLabel icon={<DatabaseOutlined />} title={t('settings.storageAndData')} description={t('settings.storageAndDataDescription')} status={<Tag>{settings.data?.results_roots?.length ?? 0}</Tag>} />,
                            children: (
                              <>
                                <Form.Item name="results_roots" label={t('settings.resultsRoots')} extra={t('settings.resultsRootsHint')} rules={[{ required: true, type: 'array', min: 1 }]}><Select mode="tags" tokenSeparators={[',']} placeholder="results/" /></Form.Item>
                                <Form.Item name="storage_monitor_path" label={t('settings.storageMonitorPath')} extra={t('settings.storageMonitorPathHint')}>
                                  <ServerDirectoryPicker placeholder={t('settings.storageMonitorPathPlaceholder')} />
                                </Form.Item>
                                <div className="form-grid-2">
                                  <Form.Item name="dataset_roots" label={t('settings.datasetRoots')} extra={t('settings.datasetRootsHint')}><Select mode="tags" tokenSeparators={[',']} placeholder="data/" /></Form.Item>
                                  <Form.Item name="managed_dataset_root" label={t('settings.managedDatasetRoot')}><Input placeholder=".alphabrain-ui/datasets" /></Form.Item>
                                </div>
                                <Form.Item name="pretrained_root" label={t('settings.pretrainedRoot')}><Input placeholder="data/pretrained_models" /></Form.Item>
                              </>
                            ),
                          },
                          {
                            key: 'remote',
                            label: (
                              <SettingsGroupLabel
                                icon={<CloudServerOutlined />}
                                title={t('settings.remoteTraining')}
                                description={t('settings.remoteTrainingSectionDescription')}
                                status={<Tag color={remoteTrainingEnabled ? 'green' : 'default'}>{t(remoteTrainingEnabled ? 'common.enabled' : 'common.disabled')}</Tag>}
                              />
                            ),
                            children: (
                              <>
                                <Alert type="info" showIcon message={t('settings.remoteTrainingHint')} />
                                <Form.Item name="remote_training_enabled" label={t('settings.remoteTrainingEnabled')} valuePropName="checked"><Switch /></Form.Item>
                                {remoteTrainingEnabled ? (
                                  <>
                                    <div className="form-grid-2">
                                      <Form.Item name="remote_training_host" label={t('settings.remoteTrainingHost')} rules={[{ required: true, whitespace: true }]}><Input placeholder="gpu-server.example.edu" /></Form.Item>
                                      <Form.Item name="remote_training_user" label={t('settings.remoteTrainingUser')}><Input placeholder="researcher" /></Form.Item>
                                      <Form.Item name="remote_training_port" label={t('settings.remoteTrainingPort')} rules={[{ required: true }]}><InputNumber min={1} max={65535} className="full-width" /></Form.Item>
                                      <Form.Item name="remote_training_repo_root" label={t('settings.remoteTrainingRepoRoot')} rules={[{ required: true, whitespace: true }]}><Input placeholder="/srv/AlphaBrain" /></Form.Item>
                                    </div>
                                    <Form.Item name="remote_training_gpu_ids" label={t('settings.remoteTrainingGpuIds')} extra={t('settings.remoteTrainingGpuIdsHint')} rules={[{ required: true, type: 'array', min: 1 }]}><Select mode="tags" tokenSeparators={[',']} placeholder="0, 1, 2, 3" /></Form.Item>
                                    <Collapse
                                      ghost
                                      className="settings-nested-collapse"
                                      items={[{
                                        key: 'ssh-advanced',
                                        label: t('settings.advancedSshOptions'),
                                        children: (
                                          <>
                                            <Form.Item name="remote_training_identity_file" label={t('settings.remoteTrainingIdentityFile')} extra={t('settings.remoteTrainingIdentityFileHint')}><Input placeholder="~/.ssh/id_ed25519" /></Form.Item>
                                            <Form.Item name="remote_training_setup_command" label={t('settings.remoteTrainingSetupCommand')} extra={t('settings.remoteTrainingSetupCommandHint')}><Input placeholder="source .venv/bin/activate" /></Form.Item>
                                          </>
                                        ),
                                      }]}
                                    />
                                  </>
                                ) : null}
                              </>
                            ),
                          },
                          {
                            key: 'runtime',
                            label: <SettingsGroupLabel icon={<ToolOutlined />} title={t('settings.localRuntime')} description={t('settings.localRuntimeDescription')} />,
                            children: (
                              <>
                                <div className="form-grid-2">
                                  <Form.Item name="cpu_utility_concurrency" label={t('settings.utilityConcurrency')}><InputNumber min={1} max={16} className="full-width" /></Form.Item>
                                  <Form.Item name="model_server_python" label={t('settings.modelServerPython')} extra={t('settings.modelServerPythonHint')}><Input placeholder="/path/to/python" /></Form.Item>
                                </div>
                                <div className="form-grid-2">
                                  <Form.Item name="low_disk_percent" label={t('settings.diskThreshold')}><InputNumber min={1} max={99} className="full-width" /></Form.Item>
                                  <Form.Item name="low_disk_gib" label={t('settings.diskThresholdGib')}><InputNumber min={0} className="full-width" /></Form.Item>
                                </div>
                              </>
                            ),
                          },
                          {
                            key: 'advanced-paths',
                            label: <SettingsGroupLabel icon={<CodeOutlined />} title={t('settings.advancedEnvironmentPaths')} description={t('settings.advancedEnvironmentPathsDescription')} />,
                            children: (
                              <>
                                <Typography.Title level={5}>{t('settings.modelDatasetPaths')}</Typography.Title>
                                <div className="form-grid-2 settings-path-grid">
                                  <Form.Item name={['environment', 'PRETRAINED_MODELS_DIR']} label="PRETRAINED_MODELS_DIR"><Input placeholder="/path/to/pretrained_models" /></Form.Item>
                                  <Form.Item name={['environment', 'LIBERO_DATA_ROOT']} label="LIBERO_DATA_ROOT"><Input placeholder="/path/to/libero" /></Form.Item>
                                  <Form.Item name={['environment', 'LEROBOT_LIBERO_DATA_DIR']} label="LEROBOT_LIBERO_DATA_DIR"><Input placeholder="/path/to/lerobot/libero" /></Form.Item>
                                  <Form.Item name={['environment', 'LIBERO_HOME']} label="LIBERO_HOME"><Input placeholder="/path/to/LIBERO" /></Form.Item>
                                  <Form.Item name={['environment', 'ROBOCASA_TABLETOP_DATA_ROOT']} label="ROBOCASA_TABLETOP_DATA_ROOT"><Input placeholder="/path/to/robocasa" /></Form.Item>
                                  <Form.Item name={['environment', 'ROBOCASA365_DATA_ROOT']} label="ROBOCASA365_DATA_ROOT"><Input placeholder="/path/to/robocasa365" /></Form.Item>
                                </div>
                                <Typography.Title level={5}>{t('settings.evaluationEnvironments')}</Typography.Title>
                                <Typography.Paragraph type="secondary">{t('settings.evaluationEnvironmentsHint')}</Typography.Paragraph>
                                <div className="form-grid-2 settings-path-grid">
                                  <Form.Item name={['environment', 'LIBERO_PYTHON']} label="LIBERO_PYTHON"><Input placeholder="/path/to/libero-env/bin/python" /></Form.Item>
                                  <Form.Item name={['environment', 'LIBERO_PLUS_HOME']} label="LIBERO_PLUS_HOME"><Input placeholder="/path/to/LIBERO-plus" /></Form.Item>
                                  <Form.Item name={['environment', 'LIBERO_PLUS_PYTHON']} label="LIBERO_PLUS_PYTHON"><Input placeholder="/path/to/libero-plus-env/bin/python" /></Form.Item>
                                  <Form.Item name={['environment', 'ROBOCASA365_PYTHON']} label="ROBOCASA365_PYTHON"><Input placeholder="/path/to/robocasa365-env/bin/python" /></Form.Item>
                                  <Form.Item name={['environment', 'ROBOCASA_TABLETOP_PYTHON']} label="ROBOCASA_TABLETOP_PYTHON"><Input placeholder="/path/to/robocasa-tabletop-env/bin/python" /></Form.Item>
                                  <Form.Item name={['environment', 'HF_HOME']} label="HF_HOME"><Input placeholder="~/.cache/huggingface" /></Form.Item>
                                  <Form.Item name={['environment', 'WANDB_BASE_URL']} label="WANDB_BASE_URL"><Input placeholder="https://api.wandb.ai" /></Form.Item>
                                  <Form.Item name={['environment', 'WANDB_MODE']} label="WANDB_MODE"><Input placeholder="online / offline / disabled" /></Form.Item>
                                </div>
                              </>
                            ),
                          },
                        ]}
                      />
                      {isAdmin ? (
                        <div className="settings-save-row">
                          <Button type="primary" htmlType="submit" icon={<SaveOutlined />} loading={updateSystem.isPending}>{t('settings.saveInfrastructure')}</Button>
                        </div>
                      ) : null}
                    </Form>
                    <Collapse
                      className="settings-groups settings-integration-group"
                      items={[{
                        key: 'wandb',
                        label: (
                          <SettingsGroupLabel
                            icon={<KeyOutlined />}
                            title={t('settings.wandbApiKeyTitle')}
                            description={t('settings.wandbSectionDescription')}
                            status={isAdmin && !wandbStatus.isLoading ? (
                              <Tag color={wandbStatus.data?.configured ? 'green' : 'default'}>
                                {t(wandbStatus.data?.configured ? 'settings.wandbConfigured' : 'settings.wandbNotConfigured')}
                              </Tag>
                            ) : null}
                          />
                        ),
                        children: (
                          <div className="wandb-secret-card">
                            <Typography.Paragraph type="secondary">{t('settings.wandbApiKeyDescription')}</Typography.Paragraph>
                            {!isAdmin ? <Alert type="info" showIcon message={t('settings.adminRequired')} /> : null}
                            {isAdmin && wandbStatus.error ? (
                              <Alert
                                type="error"
                                showIcon
                                message={t('settings.wandbStatusFailed')}
                                description={wandbStatus.error.message}
                                action={<Button size="small" onClick={() => void wandbStatus.refetch()}>{t('common.retry')}</Button>}
                              />
                            ) : null}
                            {isAdmin ? (
                              <>
                                <Input.Password
                                  value={wandbApiKey}
                                  onChange={(event) => setWandbApiKey(event.target.value)}
                                  autoComplete="new-password"
                                  visibilityToggle
                                  placeholder={wandbStatus.data?.configured ? t('settings.wandbReplacePlaceholder') : t('settings.wandbApiKeyPlaceholder')}
                                  disabled={wandbStatus.isLoading || wandbSaving || wandbDeleting}
                                  onPressEnter={() => void saveWandbApiKey()}
                                />
                                <Space wrap className="wandb-secret-actions">
                                  <Button
                                    type="primary"
                                    icon={<SaveOutlined />}
                                    loading={wandbSaving}
                                    disabled={wandbApiKey.trim().length < 8 || wandbDeleting}
                                    onClick={() => void saveWandbApiKey()}
                                  >
                                    {t(wandbStatus.data?.configured ? 'settings.wandbReplaceApiKey' : 'settings.wandbSaveApiKey')}
                                  </Button>
                                  {wandbStatus.data?.configured ? (
                                    <Button danger icon={<DeleteOutlined />} loading={wandbDeleting} disabled={wandbSaving} onClick={confirmDeleteWandbApiKey}>
                                      {t('settings.wandbDeleteApiKey')}
                                    </Button>
                                  ) : null}
                                </Space>
                              </>
                            ) : null}
                          </div>
                        ),
                      }]}
                    />
                  </div>
                ),
              },
              {
                key: 'preferences', label: <Space><GlobalOutlined />{t('settings.preferences')}</Space>,
                children: (
                  <div className="settings-panel">
                    <div className="settings-panel-heading">
                      <Typography.Title level={4}>{t('settings.preferences')}</Typography.Title>
                      <Typography.Paragraph type="secondary">{t('settings.preferencesDescription')}</Typography.Paragraph>
                    </div>
                    <Card size="small" className="settings-section-card">
                      <Form<Partial<User>> form={preferenceForm} layout="vertical" onFinish={savePreferences}>
                        <div className="form-grid-2">
                          <Form.Item name="locale" label={t('common.language')}><Select options={[{ label: '简体中文', value: 'zh-CN' satisfies Language }, { label: 'English', value: 'en-US' satisfies Language }]} /></Form.Item>
                          <Form.Item name="theme" label={t('common.theme')}><Radio.Group options={[{ label: t('common.light'), value: 'light' satisfies ThemeMode }, { label: t('common.dark'), value: 'dark' satisfies ThemeMode }]} /></Form.Item>
                        </div>
                        <Form.Item
                          name="gpu_refresh_interval_seconds"
                          label={t('settings.gpuRefreshInterval')}
                          extra={t('settings.gpuRefreshIntervalHint')}
                          rules={[{
                            validator: (_, value) => Number.isInteger(value) && (value === 0 || (value >= 2 && value <= 3600))
                              ? Promise.resolve()
                              : Promise.reject(new Error(t('settings.gpuRefreshIntervalInvalid'))),
                          }]}
                        >
                          <InputNumber min={0} max={3600} precision={0} step={1} addonAfter={t('settings.gpuRefreshSecondsUnit')} className="full-width" />
                        </Form.Item>
                        <Form.Item name="experimental_enabled" label={t('settings.userExperimental')} valuePropName="checked"><Switch disabled={!(settings.data?.experimental_allowed ?? me.data?.experimental_available)} /></Form.Item>
                        <div className="settings-save-row">
                          <Button type="primary" htmlType="submit" icon={<SaveOutlined />} loading={updatePreferences.isPending}>{t('common.save')}</Button>
                        </div>
                      </Form>
                    </Card>
                  </div>
                ),
              },
            ]}
          />
        </Card>
      </AsyncState>
    </div>
  );
}
