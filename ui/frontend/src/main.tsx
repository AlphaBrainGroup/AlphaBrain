import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider, App as AntApp, theme as antTheme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import enUS from 'antd/locale/en_US';
import { BrowserRouter } from 'react-router-dom';
import './i18n';
import './styles.css';
import { PreferencesProvider, usePreferences } from './app-context';
import { App } from './App';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 10_000 },
    mutations: { retry: false },
  },
});

function ThemedApp() {
  const { language, theme } = usePreferences();
  return (
    <ConfigProvider
      locale={language === 'zh-CN' ? zhCN : enUS}
      theme={{
        algorithm: theme === 'dark' ? antTheme.darkAlgorithm : antTheme.defaultAlgorithm,
        token: {
          colorPrimary: '#3157d5',
          colorInfo: '#3157d5',
          borderRadius: 10,
          fontFamily: "Inter, 'IBM Plex Sans', 'Noto Sans SC', system-ui, sans-serif",
        },
        components: {
          Layout: { bodyBg: theme === 'dark' ? '#0b1020' : '#f4f6fa', headerBg: theme === 'dark' ? '#111827' : '#ffffff', siderBg: theme === 'dark' ? '#111827' : '#ffffff' },
          Menu: { itemBorderRadius: 8, itemMarginInline: 10 },
          Card: { headerBg: 'transparent' },
        },
      }}
    >
      <AntApp><App /></AntApp>
    </ConfigProvider>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <PreferencesProvider>
        <BrowserRouter><ThemedApp /></BrowserRouter>
      </PreferencesProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
