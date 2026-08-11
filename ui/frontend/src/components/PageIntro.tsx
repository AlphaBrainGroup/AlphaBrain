import type { ReactNode } from 'react';
import { Typography } from 'antd';

interface PageIntroProps {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}

export function PageIntro({ title, subtitle, actions }: PageIntroProps) {
  return (
    <div className="page-intro">
      <div>
        <Typography.Title level={2}>{title}</Typography.Title>
        {subtitle ? <Typography.Text type="secondary">{subtitle}</Typography.Text> : null}
      </div>
      {actions ? <div className="page-intro-actions">{actions}</div> : null}
    </div>
  );
}
