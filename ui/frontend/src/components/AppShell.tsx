import {
  ApiOutlined,
  ApartmentOutlined,
  AuditOutlined,
  BarChartOutlined,
  CloudServerOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FundProjectionScreenOutlined,
  FileProtectOutlined,
  GlobalOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  MoonOutlined,
  PlayCircleOutlined,
  ProjectOutlined,
  SettingOutlined,
  SunOutlined,
  TeamOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Avatar, Button, Dropdown, Layout, Menu, Space, Tag, Tooltip, Typography, type MenuProps } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { Language, ThemeMode } from '../api/types';
import { usePreferences } from '../app-context';
import brandLogo from '../../logo/cropped-fav.png';

const { Header, Sider, Content } = Layout;

export function AppShell() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const { language, setLanguage, theme, setTheme } = usePreferences();
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('alphabrain-nav-collapsed') === 'true');
  const me = useQuery({ queryKey: ['me'], queryFn: api.auth.me, staleTime: 60_000 });
  const runtime = useQuery({ queryKey: ['runtime'], queryFn: api.runtime, staleTime: 60_000 });
  const logout = useMutation({
    mutationFn: api.auth.logout,
    onSuccess: () => {
      queryClient.clear();
      navigate('/login', { replace: true });
    },
  });

  useEffect(() => {
    const user = me.data;
    if (user?.locale && !localStorage.getItem('alphabrain-language')) setLanguage(user.locale);
    if (user?.theme && !localStorage.getItem('alphabrain-theme')) setTheme(user.theme);
  }, [me.data, setLanguage, setTheme]);

  const selectLanguage = (next: Language) => {
    setLanguage(next);
    void api.settings.updatePreferences({ locale: next }).catch(() => undefined);
  };
  const selectTheme = (next: ThemeMode) => {
    setTheme(next);
    void api.settings.updatePreferences({ theme: next }).catch(() => undefined);
  };

  const menuItems: MenuProps['items'] = useMemo(() => {
    const items: MenuProps['items'] = [
      { key: '/', icon: <DashboardOutlined />, label: t('nav.dashboard') },
      { key: '/experiments', icon: <ProjectOutlined />, label: t('nav.experiments') },
      { key: '/experiments/new', icon: <ExperimentOutlined />, label: t('nav.newExperiment') },
      { key: '/workloads', icon: <ApiOutlined />, label: t('nav.workloads') },
      { key: '/templates', icon: <FileProtectOutlined />, label: t('nav.templates') },
      { key: '/checkpoints', icon: <DatabaseOutlined />, label: t('nav.checkpoints') },
      { key: '/resources', icon: <CloudServerOutlined />, label: t('nav.resources') },
      { key: '/datasets', icon: <DatabaseOutlined />, label: t('nav.datasets') },
      { key: '/registry', icon: <ApartmentOutlined />, label: t('nav.registry') },
      { key: '/deployments', icon: <CloudServerOutlined />, label: t('nav.deployments') },
      { key: '/playground', icon: <PlayCircleOutlined />, label: t('nav.playground') },
      { key: '/evaluations', icon: <FundProjectionScreenOutlined />, label: t('nav.evaluations') },
      { key: '/reference-results', icon: <BarChartOutlined />, label: t('nav.referenceResults') },
      { type: 'divider' },
      { key: '/settings', icon: <SettingOutlined />, label: t('nav.settings') },
    ];
    if (me.data?.role === 'administrator') {
      items.push({ key: '/settings/users', icon: <TeamOutlined />, label: t('nav.users') });
      items.push({ key: '/settings/audit', icon: <AuditOutlined />, label: t('nav.audit') });
    }
    return items;
  }, [me.data?.role, t]);

  const activeKey = useMemo(() => {
    if (location.pathname.startsWith('/jobs') || location.pathname.startsWith('/workloads')) return '/workloads';
    if (location.pathname.startsWith('/templates')) return '/templates';
    if (location.pathname.startsWith('/checkpoints')) return '/checkpoints';
    if (location.pathname.startsWith('/resources')) return '/resources';
    if (location.pathname.startsWith('/datasets')) return '/datasets';
    if (location.pathname.startsWith('/registry')) return '/registry';
    if (location.pathname.startsWith('/deployments')) return '/deployments';
    if (location.pathname.startsWith('/playground')) return '/playground';
    if (location.pathname.startsWith('/evaluations')) return '/evaluations';
    if (location.pathname.startsWith('/evaluation-groups')) return '/evaluations';
    if (location.pathname.startsWith('/reference-results')) return '/reference-results';
    if (location.pathname === '/settings/users') return '/settings/users';
    if (location.pathname === '/settings/audit') return '/settings/audit';
    if (location.pathname.startsWith('/settings')) return '/settings';
    if (location.pathname === '/experiments/new') return '/experiments/new';
    if (location.pathname.startsWith('/experiments')) return '/experiments';
    return '/';
  }, [location.pathname]);

  const userMenu: MenuProps['items'] = [
    { key: 'profile', icon: <UserOutlined />, label: t('common.profile'), onClick: () => navigate('/settings?tab=preferences') },
  ];
  if (me.data?.deployment_mode !== 'personal') {
    userMenu.push(
      { type: 'divider' },
      { key: 'logout', icon: <LogoutOutlined />, label: t('common.logout'), danger: true, onClick: () => logout.mutate() },
    );
  }

  return (
    <Layout className="app-layout">
      <Sider
        className="app-sider"
        width={244}
        collapsedWidth={76}
        collapsed={collapsed}
        trigger={null}
      >
        <button className="brand" type="button" onClick={() => navigate('/')}>
          <span className="brand-mark"><img className="brand-logo" src={brandLogo} alt="" /></span>
          {!collapsed ? <span className="brand-copy"><b>AlphaBrain</b><small>{t('common.consoleShort').toUpperCase()}</small></span> : null}
        </button>
        <Menu
          mode="inline"
          selectedKeys={[activeKey]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
        <div className="sider-footer">
          <Button
            type="text"
            block
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => {
              setCollapsed((value) => {
                localStorage.setItem('alphabrain-nav-collapsed', String(!value));
                return !value;
              });
            }}
          >
            {!collapsed ? t('common.collapse') : null}
          </Button>
        </div>
      </Sider>
      <Layout>
        <Header className="app-header">
          {runtime.data?.demo_mode ? (
            <Tooltip title={t('common.demoModeHint')}>
              <Tag color="gold">{t('common.demoMode')}</Tag>
            </Tooltip>
          ) : null}
          <div className="header-spacer" />
          <Space size="small">
            <Tooltip title={language === 'zh-CN' ? 'English' : '简体中文'}>
              <Button
                type="text"
                icon={<GlobalOutlined />}
                onClick={() => selectLanguage(language === 'zh-CN' ? 'en-US' : 'zh-CN')}
              >
                {language === 'zh-CN' ? '中文' : 'EN'}
              </Button>
            </Tooltip>
            <Tooltip title={theme === 'light' ? t('common.dark') : t('common.light')}>
              <Button
                type="text"
                icon={theme === 'light' ? <MoonOutlined /> : <SunOutlined />}
                onClick={() => selectTheme(theme === 'light' ? 'dark' : 'light')}
                aria-label={theme === 'light' ? t('common.dark') : t('common.light')}
              />
            </Tooltip>
            <Dropdown menu={{ items: userMenu }} placement="bottomRight">
              <Button className="user-menu" type="text">
                <Avatar size={30} icon={<UserOutlined />} />
                <span className="user-copy">
                  <Typography.Text strong>{me.data?.display_name || me.data?.username}</Typography.Text>
                  <Typography.Text type="secondary">{me.data?.role === 'administrator' ? t('users.administrator') : t('users.researcher')}</Typography.Text>
                </span>
              </Button>
            </Dropdown>
          </Space>
        </Header>
        <Content className="app-content"><Outlet /></Content>
      </Layout>
    </Layout>
  );
}
