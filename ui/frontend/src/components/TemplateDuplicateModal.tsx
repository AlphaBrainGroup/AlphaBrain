import { Button, List, Modal, Space, Tag, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import type { TemplateDuplicateDetail, TemplateDuplicateMatch } from '../api/types';

interface TemplateDuplicateModalProps {
  detail?: TemplateDuplicateDetail;
  saving?: boolean;
  onCancel: () => void;
  onOpenExisting: (match: TemplateDuplicateMatch) => void;
  onSaveAnyway: () => void;
}

export function TemplateDuplicateModal({ detail, saving, onCancel, onOpenExisting, onSaveAnyway }: TemplateDuplicateModalProps) {
  const { t } = useTranslation();
  return (
    <Modal
      title={t('templates.duplicateTitle')}
      open={Boolean(detail)}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>{t('common.cancel')}</Button>,
        <Button key="save" type="primary" danger loading={saving} onClick={onSaveAnyway}>{t('templates.saveAnyway')}</Button>,
      ]}
    >
      <Typography.Paragraph type="secondary">{t('templates.duplicateDescription')}</Typography.Paragraph>
      <List
        className="template-duplicate-list"
        dataSource={detail?.matches ?? []}
        renderItem={(match) => (
          <List.Item actions={[<Button key="open" size="small" onClick={() => onOpenExisting(match)}>{t('templates.openExisting')}</Button>]}>
            <List.Item.Meta
              title={<Space size={[4, 4]} wrap><Typography.Text strong>{match.name}</Typography.Text>{match.same_name ? <Tag color="orange">{t('templates.sameName')}</Tag> : null}{match.same_spec ? <Tag color="blue">{t('templates.sameSpec')}</Tag> : null}{match.builtin ? <Tag color="geekblue">{t('templates.builtin')}</Tag> : null}</Space>}
              description={`${match.owner_name ?? t('templates.builtin')} · ${t(`templates.${match.visibility}`)}`}
            />
          </List.Item>
        )}
      />
    </Modal>
  );
}
