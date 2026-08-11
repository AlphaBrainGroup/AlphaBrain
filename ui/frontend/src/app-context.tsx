import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import i18n from './i18n';
import type { Language, ThemeMode } from './api/types';

interface PreferencesContextValue {
  language: Language;
  theme: ThemeMode;
  setLanguage: (language: Language) => void;
  setTheme: (theme: ThemeMode) => void;
}

const PreferencesContext = createContext<PreferencesContextValue | null>(null);

export function PreferencesProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(() =>
    localStorage.getItem('alphabrain-language') === 'en-US' ? 'en-US' : 'zh-CN',
  );
  const [theme, setThemeState] = useState<ThemeMode>(() =>
    localStorage.getItem('alphabrain-theme') === 'dark' ? 'dark' : 'light',
  );

  const setLanguage = useCallback((next: Language) => {
    setLanguageState(next);
    localStorage.setItem('alphabrain-language', next);
    void i18n.changeLanguage(next);
    document.documentElement.lang = next;
  }, []);

  const setTheme = useCallback((next: ThemeMode) => {
    setThemeState(next);
    localStorage.setItem('alphabrain-theme', next);
    document.documentElement.dataset.theme = next;
  }, []);

  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dataset.theme = theme;
  }, [language, theme]);

  const value = useMemo(
    () => ({ language, theme, setLanguage, setTheme }),
    [language, setLanguage, setTheme, theme],
  );
  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>;
}

export function usePreferences(): PreferencesContextValue {
  const context = useContext(PreferencesContext);
  if (!context) throw new Error('usePreferences must be used inside PreferencesProvider');
  return context;
}
