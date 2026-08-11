import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, api, healthyGpus, isRecord, listFrom, templateDuplicateDetail } from './client';
import type { DeploymentCreateRequest, EvaluationCreateRequest, ExperimentSpec, WandbRunConfig } from './types';

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

afterEach(() => vi.unstubAllGlobals());

describe('API response normalization', () => {
  it('accepts direct and paged list payloads', () => {
    expect(listFrom([1, 2])).toEqual([1, 2]);
    expect(listFrom({ items: [3], total: 1 })).toEqual([3]);
    expect(listFrom({ data: [4] })).toEqual([4]);
    expect(listFrom(undefined)).toEqual([]);
  });

  it('identifies records without treating arrays as records', () => {
    expect(isRecord({ id: 'x' })).toBe(true);
    expect(isRecord([])).toBe(false);
    expect(isRecord(null)).toBe(false);
  });

  it('parses duplicate-template conflicts and sends an explicit override', async () => {
    const detail = templateDuplicateDetail(new ApiError('duplicate_template', 409, {
      code: 'duplicate_template',
      matches: [{
        id: 'template-1', name: 'Existing', owner_name: 'Alice', visibility: 'shared',
        builtin: false, same_name: true, same_spec: false,
      }],
    }));
    expect(detail).toEqual({
      code: 'duplicate_template',
      matches: [{
        id: 'template-1', name: 'Existing', owner_id: undefined, owner_name: 'Alice',
        visibility: 'shared', builtin: false, same_name: true, same_spec: false,
      }],
    });
    expect(templateDuplicateDetail(new ApiError('conflict', 409, { code: 'other' }))).toBeUndefined();

    const spec: ExperimentSpec = {
      spec_version: 2,
      name: 'Copy',
      backbone_id: 'qwen',
      action_head_id: 'mlp',
      method_id: 'il',
      dataset_id: 'libero',
      parameters: {},
      resources: { strategy: 'auto', gpu_count: 1 },
    };
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({
      id: 'template-2', name: 'Copy', visibility: 'private', spec: {},
    }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });
    await api.templates.create({ name: 'Copy', visibility: 'personal', spec }, true);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toMatchObject({ allow_duplicate: true });
  });

  it('normalizes setup, user, and GPU backend fields', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ initialized: true, deployment_mode: 'lab' }))
      .mockResolvedValueOnce(jsonResponse({
        user: { id: 'u1', username: 'alice', display_name: 'Alice', role: 'researcher', language: 'en-US', theme: 'dark', gpu_refresh_interval_seconds: 30, is_active: true },
        experimental_available: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        available: true,
        error: 'GPU 1 memory: device lost',
        items: [
          { index: 0, name: 'RTX 4090', memory_total_bytes: 24 * 1024 ** 3, memory_used_bytes: 6 * 1024 ** 3, utilization_percent: 42, processes: [], available: true },
          { index: 1, name: 'GPU 1', memory_total_bytes: 0, memory_used_bytes: 0, utilization_percent: 0, processes: [], available: false, error: 'memory: device lost' },
        ],
      }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.setup.status()).resolves.toMatchObject({ configured: true, mode: 'laboratory' });
    await expect(api.auth.me()).resolves.toMatchObject({ username: 'alice', locale: 'en-US', theme: 'dark', gpu_refresh_interval_seconds: 30, experimental_available: true });
    const inventory = await api.gpus();
    expect(inventory).toMatchObject({
      available: true,
      error: 'GPU 1 memory: device lost',
      items: [
        { index: 0, memory_total_mb: 24576, memory_used_mb: 6144, available: true },
        { index: 1, available: false, error: 'memory: device lost' },
      ],
    });
    expect(healthyGpus(inventory)).toMatchObject([{ index: 0, name: 'RTX 4090' }]);
  });

  it('uses the backend startup suggestion before setup', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(jsonResponse({ initialized: false, deployment_mode: 'personal' })));
    await expect(api.setup.status()).resolves.toEqual({ configured: false, mode: 'personal' });
  });

  it('persists the GPU refresh interval with user preferences', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({
      id: 'u1', username: 'alice', role: 'researcher', gpu_refresh_interval_seconds: 10,
    }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    await expect(api.settings.updatePreferences({ gpu_refresh_interval_seconds: 10 })).resolves.toMatchObject({
      gpu_refresh_interval_seconds: 10,
    });
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toMatchObject({ gpu_refresh_interval_seconds: 10 });
  });

  it('inspects a dataset path and normalizes its format support', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({
      valid: true,
      path: '/datasets/collection',
      format: 'lerobot_collection',
      format_family: 'lerobot_collection',
      compatible_loaders: ['lerobot'],
      builder_support: 'mixture_only',
      builder_ready: false,
      dataset_count: 3,
      episode_count: 12,
      step_count: 240,
      parquet_count: 6,
      size_bytes: 4096,
      issues: [{ level: 'warning', code: 'collection_requires_pattern', message: 'Select child datasets with a pattern.' }],
    }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    await expect(api.datasets.inspect('/datasets/collection')).resolves.toMatchObject({
      valid: true,
      format_family: 'lerobot_collection',
      builder_support: 'mixture_only',
      builder_ready: false,
      dataset_count: 3,
      issues: [{ id: 'collection_requires_pattern', level: 'warning' }],
    });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/datasets/inspect');
    expect(init.method).toBe('POST');
    expect(JSON.parse(String(init.body))).toEqual({ path: '/datasets/collection' });
  });

  it('normalizes the capability catalog and sends the experiment envelope', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        include_experimental: false,
        catalog: {
          components: {
            backbones: [{
              id: 'qwen', kind: 'vlm', status: 'verified', label: { 'zh-CN': '千问', 'en-US': 'Qwen' },
              pretrained_directory: {
                required: true,
                configured: false,
                path: '${PRETRAINED_MODELS_DIR}/Qwen',
                exists: false,
                issue: {
                  code: 'missing_environment_variable',
                  variables: ['PRETRAINED_MODELS_DIR'],
                  message_i18n: { 'zh-CN': '环境变量未设置', 'en-US': 'Environment variable is not set' },
                },
              },
            }],
            action_heads: [{ id: 'mlp', status: 'verified', label: { 'zh-CN': '回归头', 'en-US': 'MLP' } }],
            training_methods: [{ id: 'il', status: 'verified', label: { 'zh-CN': '模仿学习', 'en-US': 'Imitation learning' } }],
            datasets: [{ id: 'libero', status: 'verified', label: { 'zh-CN': 'LIBERO', 'en-US': 'LIBERO' } }],
          },
          methods: [{
            id: 'il',
            kind: 'method',
            status: 'verified',
            name: 'Imitation learning',
            name_zh: '模仿学习',
            parameters: [{ key: 'gradient_accumulation_steps', type: 'integer', label: 'Gradient accumulation', label_zh: '梯度累积步数', default: 1, min: 1 }],
          }],
          combinations: [{
            id: 'combo', status: 'verified', backbone: 'qwen', action_head: 'mlp', method: 'il', datasets: ['libero'], min_gpus: 2,
            workflow: {
              id: 'standard', schema_version: 1,
              title: { 'zh-CN': '标准训练', 'en-US': 'Standard training' },
              fields: [{ key: 'warmup', type: 'integer', default: 10, min: 0, label: { 'zh-CN': '预热', 'en-US': 'Warmup' }, config_path: 'trainer.num_warmup_steps' }],
            },
            wandb_categories: [
              { id: 'metrics', supported: true, title: { 'zh-CN': '训练指标', 'en-US': 'Training metrics' }, description: { 'zh-CN': '标量', 'en-US': 'Scalars' } },
              { id: 'videos', supported: false, title: { 'zh-CN': '视频', 'en-US': 'Videos' }, description: { 'zh-CN': '评测视频', 'en-US': 'Evaluation videos' }, reason: { 'zh-CN': '训练器未接通', 'en-US': 'Not wired' } },
            ],
          }],
        },
      }))
      .mockResolvedValueOnce(jsonResponse({ can_submit: true, issues: [{ level: 'info', code: 'gpu_queue', message: 'Queued safely', message_i18n: { 'zh-CN': '安全排队', 'en-US': 'Queued safely' } }] }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    const catalog = await api.capabilities(false);
    expect(catalog.backbones?.[0]).toMatchObject({
      id: 'qwen',
      name: 'Qwen',
      name_zh: '千问',
      recommended_gpu_count: 2,
      pretrained_directory: {
        required: true,
        configured: false,
        path: '${PRETRAINED_MODELS_DIR}/Qwen',
        exists: false,
        issue: {
          code: 'missing_environment_variable',
          variables: ['PRETRAINED_MODELS_DIR'],
          message_i18n: { 'zh-CN': '环境变量未设置', 'en-US': 'Environment variable is not set' },
        },
      },
    });
    expect(catalog.methods?.[0].parameters).toEqual([
      expect.objectContaining({ key: 'gradient_accumulation_steps', type: 'integer', default: 1, min: 1 }),
    ]);
    expect(catalog.combinations?.[0].wandb_categories).toEqual([
      expect.objectContaining({ id: 'metrics', supported: true, title: { 'zh-CN': '训练指标', 'en-US': 'Training metrics' } }),
      expect.objectContaining({ id: 'videos', supported: false, reason: { 'zh-CN': '训练器未接通', 'en-US': 'Not wired' } }),
    ]);
    expect(catalog.combinations?.[0].workflow).toMatchObject({
      id: 'standard',
      schema_version: 1,
      fields: [expect.objectContaining({ key: 'warmup', type: 'integer', default: 10, config_path: 'trainer.num_warmup_steps' })],
    });

    const spec: ExperimentSpec = {
      name: 'smoke-run',
      backbone_id: 'qwen',
      action_head_id: 'mlp',
      method_id: 'il',
      dataset_id: 'libero',
      dataset_path: '/datasets/local-libero',
      workflow: { id: 'standard', schema_version: 1, config: { warmup: 12 } },
      resume: { mode: 'weights_only', checkpoint: '/checkpoints/base' },
      parameters: { batch_size: 4, max_steps: 100, learning_rate: 0.0001 },
      resources: { strategy: 'auto', gpu_count: 2 },
      wandb: {
        enabled: true,
        mode: 'offline',
        project: 'AlphaBrain',
        entity: 'robotics-lab',
        run_name: 'smoke-wandb',
        group: 'tests',
        job_type: 'finetune',
        tags: ['smoke'],
        notes: 'No credentials here',
        categories: ['metrics'],
        api_key: 'must-never-be-sent',
      } as WandbRunConfig & { api_key: string },
      expert_overrides: {},
    };
    await expect(api.experiments.preflight(spec)).resolves.toMatchObject({
      ok: true,
      items: [{ id: 'gpu_queue', message_i18n: { 'zh-CN': '安全排队', 'en-US': 'Queued safely' } }],
    });
    const [, init] = fetchMock.mock.calls[1] as [string, RequestInit];
    const sent = JSON.parse(String(init.body)) as Record<string, unknown>;
    expect(sent).toMatchObject({
      name: 'smoke-run',
      acknowledge_experimental: false,
      spec: {
        spec_version: 2,
        architecture: { backbone: 'qwen', action_head: 'mlp' },
        training: { method: 'il' },
        dataset: { id: 'libero', root: '/datasets/local-libero', data_root: '/datasets/local-libero' },
        parameters: { run_id: 'smoke-run', per_device_batch_size: 4, max_train_steps: 100 },
        resources: { allocation: 'auto', num_gpus: 2 },
        workflow: { id: 'standard', schema_version: 1, config: { warmup: 12 } },
        resume: { mode: 'weights_only', checkpoint: '/checkpoints/base' },
        wandb: {
          enabled: true,
          mode: 'offline',
          project: 'AlphaBrain',
          entity: 'robotics-lab',
          run_name: 'smoke-wandb',
          group: 'tests',
          job_type: 'finetune',
          tags: ['smoke'],
          notes: 'No credentials here',
          categories: ['metrics'],
        },
      },
    });
    expect(JSON.stringify(sent)).not.toContain('must-never-be-sent');
    expect(JSON.stringify(sent)).not.toContain('api_key');
  });

  it('keeps the W&B API Key on the dedicated write-only settings endpoint', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        deployment_mode: 'personal',
        environment: { PRETRAINED_MODELS_DIR: '/models', WANDB_API_KEY: 'legacy-secret-that-must-be-dropped' },
      }))
      .mockResolvedValueOnce(jsonResponse({
        deployment_mode: 'personal',
        environment: { PRETRAINED_MODELS_DIR: '/new-models' },
      }))
      .mockResolvedValueOnce(jsonResponse({ configured: false }))
      .mockResolvedValueOnce(jsonResponse({ configured: true }))
      .mockResolvedValueOnce(jsonResponse({ configured: false }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    await expect(api.settings.get()).resolves.toMatchObject({
      environment: { PRETRAINED_MODELS_DIR: '/models' },
    });
    await api.settings.update({
      environment: { PRETRAINED_MODELS_DIR: '/new-models', WANDB_API_KEY: 'must-not-use-general-settings' },
    });
    await expect(api.settings.wandb.status()).resolves.toEqual({ configured: false });
    await expect(api.settings.wandb.setApiKey('dedicated-secret')).resolves.toEqual({ configured: true });
    await expect(api.settings.wandb.deleteApiKey()).resolves.toEqual({ configured: false });

    const [settingsUrl, settingsInit] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(settingsUrl).toContain('/settings');
    expect(JSON.parse(String(settingsInit.body))).toEqual({ environment: { PRETRAINED_MODELS_DIR: '/new-models' } });
    const [statusUrl, statusInit] = fetchMock.mock.calls[2] as [string, RequestInit];
    expect(statusUrl).toContain('/settings/wandb');
    expect(statusInit.method).toBeUndefined();
    const [setUrl, setInit] = fetchMock.mock.calls[3] as [string, RequestInit];
    expect(setUrl).toContain('/settings/wandb/api-key');
    expect(setInit.method).toBe('PUT');
    expect(JSON.parse(String(setInit.body))).toEqual({ api_key: 'dedicated-secret' });
    const [deleteUrl, deleteInit] = fetchMock.mock.calls[4] as [string, RequestInit];
    expect(deleteUrl).toContain('/settings/wandb/api-key');
    expect(deleteInit.method).toBe('DELETE');
    expect(deleteInit.body).toBeUndefined();
  });

  it('round-trips credential-free W&B preferences through templates', async () => {
    const templateSpec = {
      architecture: { backbone: 'qwen', action_head: 'mlp' },
      training: { method: 'il' },
      dataset: { id: 'libero' },
      parameters: { run_id: 'wandb-template', per_device_batch_size: 4 },
      resources: { allocation: 'auto', num_gpus: 1 },
      wandb: {
        enabled: true,
        mode: 'online',
        project: 'robotics',
        entity: 'lab',
        run_name: 'baseline',
        group: 'ablation',
        job_type: 'finetune',
        tags: ['qwen', 'libero'],
        notes: 'Template metadata',
        categories: ['metrics', 'config', 'system'],
      },
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse([{ id: 't1', name: 'W&B template', visibility: 'private', spec: templateSpec }]))
      .mockResolvedValueOnce(jsonResponse({ id: 't2', name: 'W&B copy', visibility: 'private', spec: templateSpec }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    const [template] = await api.templates.list();
    expect(template.spec?.wandb).toEqual(templateSpec.wandb);
    await api.templates.create({ name: 'W&B copy', visibility: 'personal', spec: template.spec });

    const [, init] = fetchMock.mock.calls[1] as [string, RequestInit];
    const sent = JSON.parse(String(init.body)) as Record<string, unknown>;
    expect(sent).toMatchObject({ spec: { wandb: templateSpec.wandb } });
    expect(JSON.stringify(sent)).not.toContain('api_key');
  });

  it('normalizes the evaluation registry and preserves the v1 request and result contracts', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        presets: [{ id: 'quick', label: { 'zh-CN': '快速检查', 'en-US': 'Quick check' }, description: { 'zh-CN': '一个 episode', 'en-US': 'One episode' }, episodes_per_task: 1, task_limit: 1 }],
        benchmarks: [{
          id: 'libero', status: 'verified', label: { 'zh-CN': 'LIBERO 中文', 'en-US': 'LIBERO' },
          description: { 'zh-CN': '仿真评测', 'en-US': 'Simulation evaluation' },
          suites: [{ id: 'libero_goal', label: { 'zh-CN': '目标', 'en-US': 'Goal' } }],
          default_suite: 'libero_goal',
          environment: [{ key: 'LIBERO_HOME', required: true, configured: true, available: true }],
          readiness: { ready: true },
          compatibility: { combination_ids: ['qwen-oft'] },
          parameter_schema: { type: 'object', properties: { num_trials: { type: 'integer', default: 1, minimum: 1, title: { 'zh-CN': '次数', 'en-US': 'Trials' } } } },
        }],
      }))
      .mockResolvedValueOnce(jsonResponse({ can_submit: true, issues: [{ level: 'info', code: 'static_preflight_passed', message: 'Ready' }] }))
      .mockResolvedValueOnce(jsonResponse({
        schema_version: 'evaluation-result-v1', status: 'completed', benchmark: 'libero', checkpoint: '/checkpoints/model',
        suite: { id: 'libero_goal' }, duration_seconds: 12.5,
        summary: { num_tasks: 1, num_episodes: 2, num_successes: 1, success_rate: 0.5 },
        tasks: [{ id: 'task-1', name: 'Task one', num_episodes: 2, num_successes: 1, success_rate: 0.5 }],
        episodes: [{ task_id: 'task-1', task_name: 'Task one', episode_index: 0, success: true, steps: 42, video: 'videos/task-1.mp4' }],
        videos: [{ path: 'videos/task-1.mp4', task_id: 'task-1', task_name: 'Task one', episode_index: 0, success: true }],
      }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    const catalog = await api.evaluations.benchmarks();
    expect(catalog.benchmarks[0]).toMatchObject({
      id: 'libero', name: 'LIBERO', name_zh: 'LIBERO 中文', available: true,
      defaults: { suite: 'libero_goal' },
      suites: [{ value: 'libero_goal', label: 'Goal', label_zh: '目标' }],
      compatibility: { combination_ids: ['qwen-oft'] },
      parameters: [{ key: 'num_trials', type: 'integer', default: 1, min: 1, label: 'Trials', label_zh: '次数' }],
    });
    expect(catalog.benchmarks[0].presets[0]).toMatchObject({ id: 'quick', name: 'Quick check', name_zh: '快速检查' });

    const payload: EvaluationCreateRequest = {
      name: 'LIBERO smoke',
      checkpoint_source: { kind: 'indexed', checkpoint_id: 'checkpoint-1' },
      combination_id: 'qwen-oft', benchmark_id: 'libero', preset: 'quick', suite: 'libero_goal',
      parameters: {}, resources: { strategy: 'auto', gpu_count: 1 },
      wandb: { enabled: true, mode: 'online', project: 'evaluation', entity: '', run_name: '', tags: [], categories: ['summary', 'tasks', 'config'] },
      acknowledge_experimental: false,
    };
    await expect(api.evaluations.preflight(payload)).resolves.toMatchObject({ ok: true });
    const [, preflightInit] = fetchMock.mock.calls[1] as [string, RequestInit];
    const sent = JSON.parse(String(preflightInit.body));
    expect(sent).toMatchObject({ suite: 'libero_goal', parameters: {}, wandb: { categories: ['summary', 'tasks', 'config'] } });
    expect(sent.parameters).not.toHaveProperty('suite');

    await expect(api.evaluations.result('evaluation-1')).resolves.toMatchObject({
      schema_version: 'evaluation-result-v1', benchmark_id: 'libero', duration_seconds: 12.5,
      summary: { num_tasks: 1, num_episodes: 2, num_successes: 1, success_rate: 0.5 },
      tasks: [{ id: 'task-1', num_episodes: 2, num_successes: 1, success_rate: 0.5 }],
      episodes: [{ task_id: 'task-1', episode_index: 0, success: true }],
      videos: [{ task_id: 'task-1', episode_index: 0, success: true }],
    });
  });

  it('does not turn an empty evaluation result summary into a zero-percent result', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse([
      {
        id: 'queued-evaluation',
        name: 'Queued evaluation',
        status: 'queued',
        benchmark_id: 'libero',
        result_summary: {},
      },
      {
        id: 'running-evaluation',
        name: 'Running evaluation',
        status: 'running',
        benchmark_id: 'libero',
        result_summary: {},
      },
      {
        id: 'completed-evaluation',
        name: 'Completed evaluation',
        status: 'completed',
        benchmark_id: 'libero',
        result_summary: {},
      },
    ]));
    vi.stubGlobal('fetch', fetchMock);

    const runs = await api.evaluations.list();
    expect(runs).toHaveLength(3);
    for (const run of runs) expect(run.result).toBeUndefined();
  });

  it('normalizes checkpoint inspection candidates for evaluation creation', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({
      ok: false,
      detected: {
        framework: 'QwenOFT',
        backbone: 'qwen2_5_vl',
        action_head: 'oft',
        candidate_combination_ids: ['qwen-oft', 'qwen-oft-experimental'],
      },
      candidates: [
        { combination_id: 'qwen-oft', benchmark_ids: ['libero'] },
        { combination_id: 'qwen-oft-experimental', benchmark_ids: ['libero', 'libero_plus'] },
      ],
      compatible_benchmark_ids: [],
      issues: [{
        severity: 'error',
        code: 'deployment_combination_ambiguous',
        field: 'combination_id',
        message_i18n: { 'zh-CN': '请选择组合', 'en-US': 'Select a combination' },
      }],
    }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    const source = { kind: 'indexed', checkpoint_id: 'checkpoint-1' } as const;
    await expect(api.evaluations.inspectCheckpoint(source, 'qwen-oft')).resolves.toEqual({
      valid: false,
      detected: {
        framework: 'QwenOFT',
        backbone: 'qwen2_5_vl',
        action_head: 'oft',
        adapter_id: undefined,
        combination_id: undefined,
        candidate_combination_ids: ['qwen-oft', 'qwen-oft-experimental'],
      },
      evaluation_candidates: [
        { combination_id: 'qwen-oft', benchmark_ids: ['libero'] },
        { combination_id: 'qwen-oft-experimental', benchmark_ids: ['libero', 'libero_plus'] },
      ],
      compatible_benchmark_ids: [],
      issues: [expect.objectContaining({ id: 'deployment_combination_ambiguous', level: 'error', path: 'combination_id' })],
    });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/evaluations/inspect-checkpoint');
    expect(init.method).toBe('POST');
    expect(JSON.parse(String(init.body))).toEqual({ ...source, combination_id: 'qwen-oft' });
  });

  it('preserves ownership and dashboard alerts and uses DELETE for user removal', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        job_counts: {},
        recent_experiments: [],
        alerts: [{ level: 'warning', code: 'low_disk_space', path: '/results' }],
        system_metrics: {
          available: true,
          cpu_percent: 24.6,
          load: { one_minute: 1.2, five_minutes: 0.8, fifteen_minutes: 0.4 },
          memory: { total_bytes: 1000, used_bytes: 400, available_bytes: 600, percent: 40 },
        },
        remote_training: { enabled: true, target: 'researcher@gpu.example.edu' },
      }))
      .mockResolvedValueOnce(jsonResponse([{ path: '/results', mount_point: '/mnt/results', used_bytes: 90, total_bytes: 100, free_bytes: 10, low_space: true }]))
      .mockResolvedValueOnce(jsonResponse([{ id: 'j1', experiment_id: 'e1', owner_id: 'u1', name: 'Run', status: 'queued', requested_gpu_ids: [], assigned_gpu_ids: [] }]))
      .mockResolvedValueOnce(jsonResponse([{ id: 't1', owner_id: 'u1', name: 'Template', visibility: 'shared', spec: { parameters: { run_id: 'old-run', per_device_batch_size: 4, gradient_accumulation_steps: 2 }, resources: {} } }]))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    await expect(api.dashboard()).resolves.toMatchObject({
      alerts: [{ id: 'low_disk_space', level: 'warning', message: '/results' }],
      storage: { path: '/results', mount_point: '/mnt/results', warning: true },
      system_metrics: {
        available: true,
        cpu_percent: 24.6,
        load: { one_minute: 1.2, five_minutes: 0.8, fifteen_minutes: 0.4 },
        memory: { total_bytes: 1000, used_bytes: 400, available_bytes: 600, percent: 40 },
      },
      remote_training: { enabled: true, target: 'researcher@gpu.example.edu' },
    });
    await expect(api.jobs.list()).resolves.toMatchObject([{ id: 'j1', owner_id: 'u1' }]);
    await expect(api.templates.list()).resolves.toMatchObject([{
      id: 't1',
      owner_id: 'u1',
      spec: { parameters: { batch_size: 4, gradient_accumulation_steps: 2 } },
    }]);
    await api.users.remove('u1');

    const [, init] = fetchMock.mock.calls[4] as [string, RequestInit];
    expect(init.method).toBe('DELETE');
  });

  it('normalizes remote training server metrics', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({
      enabled: true,
      available: true,
      target: 'researcher@gpu.example.edu',
      hostname: 'gpu-node-01',
      collected_at: '2026-07-29T08:00:00Z',
      gpus: [{
        index: 2,
        uuid: 'GPU-2',
        name: 'RTX 4090',
        memory_total_bytes: 24 * 1024 ** 3,
        memory_used_bytes: 8 * 1024 ** 3,
        utilization_percent: 60,
        processes: [],
        available: true,
      }],
      system_metrics: {
        available: true,
        cpu_percent: 30,
        memory: { total_bytes: 1000, used_bytes: 500, available_bytes: 500, percent: 50 },
      },
      storage: { path: '/srv/AlphaBrain', used_bytes: 250, total_bytes: 1000, free_bytes: 750 },
    }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.remoteTraining.metrics()).resolves.toMatchObject({
      enabled: true,
      available: true,
      hostname: 'gpu-node-01',
      gpus: [{ index: 2, memory_total_mb: 24576, memory_used_mb: 8192, available: true }],
      system_metrics: { cpu_percent: 30, memory: { percent: 50 } },
      storage: { path: '/srv/AlphaBrain', free_bytes: 750 },
    });
  });

  it('normalizes the configured training target without exposing SSH settings', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse({
      mode: 'remote',
      gpu_ids: [0, 1, 2, 3, 3, -1, '4'],
    }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.trainingTarget()).resolves.toEqual({
      mode: 'remote',
      gpu_ids: [0, 1, 2, 3],
    });
  });

  it('accepts the dashboard system compatibility alias and structured telemetry errors', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        job_counts: {},
        system: {
          available: false,
          cpu_percent: null,
          load: null,
          memory: null,
          error: { code: 'system_metrics_unavailable', message: 'permission denied' },
        },
      }))
      .mockResolvedValueOnce(jsonResponse([]));
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.dashboard()).resolves.toMatchObject({
      system_metrics: {
        available: false,
        error: { code: 'system_metrics_unavailable', message: 'permission denied' },
      },
      system: {
        available: false,
        error: { code: 'system_metrics_unavailable', message: 'permission denied' },
      },
    });
  });

  it('normalizes deployment registry JSON Schema and deployment lifecycle responses', async () => {
    const runningDeployment = {
      id: 'd1', name: 'Policy server', owner_id: 'u1', owner_name: 'Alice', status: 'running',
      combination_id: 'deploy_cosmos', adapter_id: 'cosmos_policy_websocket', backbone_id: 'cosmos2', action_head_id: 'cosmos_dit',
      requested_gpu_count: 2, requested_gpu_ids: [], assigned_gpu_ids: [0, 1], parameters: { pretrained_dir: '/models/cosmos' },
      endpoint_scope: 'lan', bind_host: '0.0.0.0', advertised_host: 'gpu.local', port: 10093,
      endpoint: { scope: 'lan', bind_host: '0.0.0.0', advertised_host: 'gpu.local', port: 10093, url: 'ws://gpu.local:10093' },
      idle_timeout_seconds: 1800, api_key_prefix: 'ab_1234', command_preview: 'python server.py',
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        catalog: {
          adapters: [{
            id: 'cosmos_policy_websocket',
            label: { 'zh-CN': 'Cosmos 模型服务', 'en-US': 'Cosmos model server' },
            protocol: { transport: 'websocket', request_format: 'alphabrain_policy_v1' },
            parameter_schema: {
              required: ['pretrained_dir'],
              properties: {
                pretrained_dir: { type: 'string', format: 'local-directory', title: { 'zh-CN': '基础模型目录', 'en-US': 'Base model directory' } },
                port: { type: 'integer', default: 10093 },
                idle_timeout_seconds: { type: 'integer', default: 1800 },
                use_bf16: { type: 'boolean', default: true },
              },
            },
          }],
          combinations: [{
            id: 'deploy_cosmos', status: 'verified', backbone: 'cosmos2', action_head: 'cosmos_dit', adapter: 'cosmos_policy_websocket',
            parameter_schema: {
              properties: {
                guidance_scale: { type: 'number', default: 1.5, title: { 'zh-CN': '引导强度', 'en-US': 'Guidance scale' } },
              },
            },
          }],
          components: {
            backbones: [{ id: 'cosmos2', label: { 'zh-CN': '宇宙世界模型', 'en-US': 'Cosmos world model' } }],
            action_heads: [{ id: 'cosmos_dit', label: { 'zh-CN': 'Cosmos 动作头', 'en-US': 'Cosmos action head' } }],
          },
        },
        include_experimental: true,
        experimental_allowed: true,
      }))
      .mockResolvedValueOnce(jsonResponse([runningDeployment]))
      .mockResolvedValueOnce(jsonResponse({
        ok: true, can_submit: true,
        items: [{ severity: 'success', level: 'success', code: 'deployment_preflight_passed', message: '部署预检通过', message_i18n: { 'zh-CN': '部署预检通过', 'en-US': 'Deployment preflight passed' } }],
        issues: [], resolved: { combination_id: 'deploy_cosmos' },
      }))
      .mockResolvedValueOnce(jsonResponse({ deployment: { ...runningDeployment, id: 'd2', name: 'New server', status: 'queued', assigned_gpu_ids: [], endpoint: { ...runningDeployment.endpoint, port: null, url: null }, port: null }, api_key: 'only-once' }))
      .mockResolvedValueOnce(jsonResponse({ deployment: { ...runningDeployment, status: 'queued', assigned_gpu_ids: [] } }))
      .mockResolvedValueOnce(jsonResponse({ deployment: runningDeployment, api_key: 'rotated-once' }))
      .mockResolvedValueOnce(jsonResponse({ ...runningDeployment, status: 'stopping' }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    const catalog = await api.deployments.capabilities();
    expect(catalog.adapters[0]).toMatchObject({
      id: 'cosmos_policy_websocket',
      protocol: 'alphabrain_policy_v1',
      parameters: [
        expect.objectContaining({ key: 'pretrained_dir', type: 'path', required: true }),
        expect.objectContaining({ key: 'use_bf16', type: 'boolean', default: true }),
      ],
    });
    expect(catalog.adapters[0].parameters.map((item) => item.key)).not.toContain('port');
    expect(catalog.experimental_allowed).toBe(true);
    expect(catalog.combinations[0]).toMatchObject({
      name: 'Cosmos world model + Cosmos action head',
      name_zh: '宇宙世界模型 + Cosmos 动作头',
      parameters: [expect.objectContaining({ key: 'guidance_scale', type: 'number', default: 1.5 })],
    });
    await expect(api.deployments.list()).resolves.toMatchObject([{
      id: 'd1', status: 'running', assigned_gpu_ids: [0, 1], endpoint: { scope: 'lan', url: 'ws://gpu.local:10093' },
    }]);

    const request: DeploymentCreateRequest = {
      name: 'New server', checkpoint_source: { kind: 'local', path: '/checkpoints/run-1' }, combination_id: 'deploy_cosmos',
      resources: { strategy: 'auto', gpu_count: 1 },
      endpoint: { scope: 'lan', advertised_host: 'gpu.local', idle_timeout_seconds: 1800 },
      parameters: { pretrained_dir: '/models/cosmos' },
    };
    await expect(api.deployments.preflight(request)).resolves.toMatchObject({
      ok: true,
      items: [{ id: 'deployment_preflight_passed', level: 'success', message_i18n: { 'en-US': 'Deployment preflight passed' } }],
    });
    await expect(api.deployments.create(request)).resolves.toMatchObject({ deployment: { id: 'd2', status: 'queued' }, api_key: 'only-once' });
    const [, init] = fetchMock.mock.calls[3] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual(request);
    await expect(api.deployments.restart('d1')).resolves.toMatchObject({ deployment: { id: 'd1', status: 'queued' }, api_key: undefined });
    await expect(api.deployments.rotateKey('d1')).resolves.toMatchObject({ deployment: { id: 'd1', status: 'running' }, api_key: 'rotated-once' });
    await expect(api.deployments.stop('d1')).resolves.toMatchObject({ id: 'd1', status: 'stopping' });
    expect(fetchMock.mock.calls.slice(4).map(([url]) => url)).toEqual([
      '/api/v1/deployments/d1/restart',
      '/api/v1/deployments/d1/rotate-key',
      '/api/v1/deployments/d1/stop',
    ]);
  });

  it('round-trips dedicated model-server Python and storage-monitor settings', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ deployment_mode: 'personal', results_roots: ['results', '/mnt/lab-results'], storage_monitor_path: '/mnt/models', model_server_python: '/opt/alphabrain/bin/python', secure_cookies: false }))
      .mockResolvedValueOnce(jsonResponse({ deployment_mode: 'personal', results_roots: ['results', '/mnt/lab-results'], storage_monitor_path: '/mnt/datasets', model_server_python: '/srv/model/bin/python', secure_cookies: true }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    await expect(api.settings.get()).resolves.toMatchObject({ model_server_python: '/opt/alphabrain/bin/python', results_roots: ['results', '/mnt/lab-results'], storage_monitor_path: '/mnt/models', secure_cookies: false });
    await expect(api.settings.update({ model_server_python: '/srv/model/bin/python', results_roots: ['results', '/mnt/lab-results'], storage_monitor_path: '/mnt/datasets', secure_cookies: true })).resolves.toMatchObject({ model_server_python: '/srv/model/bin/python', storage_monitor_path: '/mnt/datasets', secure_cookies: true });
    const [, init] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({ model_server_python: '/srv/model/bin/python', results_roots: ['results', '/mnt/lab-results'], storage_monitor_path: '/mnt/datasets', secure_cookies: true });
  });

  it('uses the safe registry overlay and local reference-result endpoints', async () => {
    const overlay = {
      schema: 'alphabrain.registry-overlay',
      schema_version: 1,
      overlay_version: 'lab-v1',
      additions: { components: { backbones: [], action_heads: [], training_methods: [], datasets: [] }, combinations: [], deployment_combinations: [] },
      disables: { components: { backbones: [], action_heads: [], training_methods: [], datasets: [] }, combinations: [], deployment_combinations: [] },
    };
    const view = {
      schema: 'alphabrain.registry-view', schema_version: 1, registry_version: 'test',
      components: [], combinations: [], deployment_adapters: [], deployment_combinations: [], workflow_ids: [], warnings: [],
      summary: { component_count: 0, combination_count: 0, overlay_component_count: 0, overlay_combination_count: 0, deployment_adapter_count: 0, deployment_combination_count: 0, overlay_deployment_combination_count: 0 },
    };
    const status = { configured: false, sha256: null, overlay_version: null, component_additions: 0, combination_additions: 0, deployment_combination_additions: 0, disabled_entries: 0 };
    const snapshot = {
      schema: 'alphabrain.reference-results', schema_version: 1, version: '2026-07-16',
      source_url: 'https://www.alphabrain-platform.com/#features', source_asset: 'https://www.alphabrain-platform.com/assets/app.js',
      captured_at: '2026-07-16T00:00:00+08:00', title: 'Reference', disclaimer: 'Not reproduced', result_sets: [],
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ view, overlay_status: status }))
      .mockResolvedValueOnce(jsonResponse({ status, overlay: null }))
      .mockResolvedValueOnce(jsonResponse({ overlay, view }))
      .mockResolvedValueOnce(jsonResponse({ status: { ...status, configured: true }, overlay }))
      .mockResolvedValueOnce(jsonResponse({ status, overlay: null }))
      .mockResolvedValueOnce(jsonResponse(snapshot))
      .mockResolvedValueOnce(jsonResponse({ schema: 'alphabrain.reference-results-match', schema_version: 1, snapshot, benchmark_id: 'libero', signature: { suite: 'libero_all' }, items: [] }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('document', { cookie: '' });

    await expect(api.registry.browse()).resolves.toMatchObject({ view: { registry_version: 'test' } });
    await expect(api.registry.overlay.get()).resolves.toMatchObject({ status: { configured: false } });
    await expect(api.registry.overlay.preview(overlay)).resolves.toMatchObject({ overlay: { overlay_version: 'lab-v1' } });
    await expect(api.registry.overlay.save(overlay)).resolves.toMatchObject({ status: { configured: true } });
    await expect(api.registry.overlay.remove()).resolves.toMatchObject({ status: { configured: false } });
    await expect(api.referenceResults.list()).resolves.toMatchObject({ version: '2026-07-16' });
    await expect(api.referenceResults.match('libero', { suite: 'libero_all' })).resolves.toMatchObject({ items: [] });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      '/api/v1/registry',
      '/api/v1/registry/overlay',
      '/api/v1/registry/overlay/preview',
      '/api/v1/registry/overlay',
      '/api/v1/registry/overlay',
      '/api/v1/reference-results',
      '/api/v1/reference-results/match',
    ]);
    expect(JSON.parse(String((fetchMock.mock.calls[2][1] as RequestInit).body))).toEqual(overlay);
    expect(JSON.parse(String((fetchMock.mock.calls[6][1] as RequestInit).body))).toEqual({
      benchmark_id: 'libero',
      signature: { suite: 'libero_all' },
    });
  });
});
