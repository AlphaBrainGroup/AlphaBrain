import { Card, Checkbox, Form, Input, InputNumber, Select, Space, Tag, Typography } from 'antd';
import type { Language, WorkflowField, WorkflowSchema } from '../api/types';

function localized(
  value: Partial<Record<Language, string>> | undefined,
  language: Language,
  fallback: string,
): string {
  return value?.[language] ?? value?.['en-US'] ?? value?.['zh-CN'] ?? fallback;
}

function WorkflowInput({ field, language }: { field: WorkflowField; language: Language }) {
  if (field.type === 'boolean') return <Checkbox />;
  if (field.type === 'select') {
    return (
      <Select
        options={(field.options ?? []).map((option) => ({
          value: option.value,
          label: localized(option.label, language, String(option.value)),
        }))}
      />
    );
  }
  if (field.type === 'integer' || field.type === 'number') {
    return (
      <InputNumber
        className="full-width"
        min={field.min}
        max={field.max}
        precision={field.type === 'integer' ? 0 : undefined}
      />
    );
  }
  if (field.type === 'textarea') return <Input.TextArea rows={3} />;
  return <Input placeholder={field.type === 'path' ? '/path/to/...' : undefined} />;
}

export function TrainingWorkflowFields({
  schema,
  fields,
  language,
}: {
  schema: WorkflowSchema;
  fields: WorkflowField[];
  language: Language;
}) {
  if (!schema.fields.length) return null;
  return (
    <Card
      size="small"
      title={localized(schema.title, language, schema.id)}
      extra={<Tag color="blue">spec v2 · schema {schema.schema_version}</Tag>}
    >
      <Typography.Paragraph type="secondary">
        {localized(schema.description, language, '')}
      </Typography.Paragraph>
      <div className="form-grid-2">
        {fields.map((field) => (
          <Form.Item
            key={field.key}
            name={['workflow_config', field.key]}
            label={(
              <Space size={6} wrap>
                <span>{localized(field.label, language, field.key)}</span>
                {field.stage ? <Tag>{field.stage}</Tag> : null}
              </Space>
            )}
            tooltip={localized(field.description, language, '') || undefined}
            extra={field.config_path ? <Typography.Text type="secondary" code>{field.config_path}</Typography.Text> : undefined}
            valuePropName={field.type === 'boolean' ? 'checked' : 'value'}
            rules={[{ required: field.required }]}
          >
            <WorkflowInput field={field} language={language} />
          </Form.Item>
        ))}
      </div>
    </Card>
  );
}
